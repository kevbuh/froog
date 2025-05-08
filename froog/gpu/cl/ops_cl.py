#  _______  ______    _______  _______  _______ 
# |       ||    _ |  |       ||       ||       |
# |    ___||   | ||  |   _   ||   _   ||    ___|
# |   |___ |   |_||_ |  | |  ||  | |  ||   | __ 
# |    ___||    __  ||  |_|  ||  |_|  ||   ||  |
# |   |    |   |  | ||       ||       ||   |_| |
# |___|    |___|  |_||_______||_______||_______|
#
# OpenCL kernels

import numpy as np
from typing import Any, Tuple, Union, List, Optional, Dict, Callable
from ...tensor import Function, register
import pyopencl as cl
from .cl_utils import (
    get_size, buffer_new, buffer_zeros, buffer_like, 
    clbuild, binary_op, unary_op, cl_pooling_krnl_build, pooling_op
)

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    return binary_op(ctx, 'a+b', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add, gpu=True)

class Sub(Function):
  @staticmethod
  def forward(ctx, x, y):
    return binary_op(ctx, 'a-b', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    not_grad_output = unary_op(ctx, '-a', grad_output)
    return grad_output, not_grad_output
register('sub', Sub, gpu=True)

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(ctx, 'a*b', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return binary_op(ctx, 'a*b', y, grad_output), binary_op(ctx, 'a*b', x, grad_output)
register('mul', Mul, gpu=True)

class Pow(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(ctx, 'pow(a,b)', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x = binary_op(ctx, 'a*b', grad_output, binary_op(ctx, 'b * (pow((float)a, (float)(b-1.0)));', x, y))
    grad_y = binary_op(ctx, 'a*b', grad_output, binary_op(ctx, 'pow((float)a, (float)b) * log(a);', x, y))
    return grad_x, grad_y
register('pow', Pow, gpu=True)

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    ret = buffer_new(ctx, (1,)) # buffer of size 1, which will hold the sum
    prg = clbuild(ctx.cl_ctx, """
    __kernel void sum(
        __global const float *a_g, int sz, __global float *res_g)
    {
      float out = 0.0;
      for (int x = 0; x < sz; x++) {
        out += a_g[x];
      }
      res_g[0] = out;
    }
    """)
    prg.sum(ctx.cl_queue, [input.shape[0]], None, input, np.int32(get_size(input)), ret)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    ret = buffer_like(ctx, input)
    prg = clbuild(ctx.cl_ctx, """
    __kernel void fill(
        __global const float *a_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[0];
    }
    """)
    prg.fill(ctx.cl_queue, [get_size(ret)], None, grad_output, ret)
    return ret
register('sum', Sum, gpu=True)

class Dot(Function):
  """
  A[gid_y * size + i] accesses an element in the row gid_y of matrix A and the column i
  """
  @staticmethod
  def forward(ctx, input, weight):
    assert input.shape[1] == weight.shape[0] # inner dims must match for dot product
    isize = np.int32(input.shape[0])
    msize = np.int32(input.shape[1])
    osize = np.int32(weight.shape[1])
    one = np.int32(1)
    ret = buffer_new(ctx, (isize, osize))
    prg = clbuild(ctx.cl_ctx, """
    __kernel void matmul(
        __global const float *input,
        __global const float *weight,
        __global float *res,
        int input_row_size,
        int input_col_size,
        int msize,
        int weight_row_size,
        int weight_col_size,
        int osize
        )
    {
      int gid_y = get_global_id(0); // row index
      int gid_x = get_global_id(1); // col index
      
      float acc = 0.0;
      for (int i = 0; i < msize; i++) {
        acc += input[gid_y * input_row_size + i * input_col_size] * weight[gid_x * weight_row_size + i * weight_col_size];
      }
      res[gid_y * osize + gid_x] = acc;
    }
    """)
    ctx.save_for_backward(input, weight, prg)
    # (isize,msize) x (msize,osize) = (isize,osize)
    prg.matmul(ctx.cl_queue, [isize, osize], None,
      input, weight, ret,
      msize, one, msize, one, osize, osize)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    input, weight, prg = ctx.saved_tensors
    isize = np.int32(input.shape[0])
    msize = np.int32(input.shape[1])
    osize = np.int32(weight.shape[1])
    one = np.int32(1)

    grad_input = buffer_like(ctx, input)
    grad_weight = buffer_like(ctx, weight)

    # (isize,osize) x (msize,osize) = (isize,msize)
    prg.matmul(ctx.cl_queue, [isize, msize], None,
      grad_output, weight, grad_input,
      osize, one, osize, osize, one, msize)

    # (isize,msize) x (isize,osize) = (msize,osize)
    prg.matmul(ctx.cl_queue, [msize, osize], None,
      input, grad_output, grad_weight,
      one, msize, isize, one, osize, osize)

    return grad_input, grad_weight
register('dot', Dot, gpu=True)
register('matmul', Dot, gpu=True)

# ***** SIMPLE OPS ********

class Reshape(Function):
  @staticmethod
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    dims = list(shape)
    product_of_specified_dims = 1
    for dim in dims:
      if dim != -1: 
        product_of_specified_dims *= dim
    for i, dim in enumerate(dims):
      if dim == -1:
        dims[i] = get_size(x) // product_of_specified_dims # calculate missing dimension
    assert get_size(x) == np.prod(dims)
    x.shape = tuple(dims)
    return x

  @staticmethod
  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    grad_output.shape = in_shape
    return grad_output
register('reshape', Reshape, gpu=True)

# ***** ACTIVATION OPS ********

class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op(ctx, 'max(a, (float)0.);', input)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(ctx, 'a * (b >= 0);', grad_output, input)
register('relu', ReLU, gpu=True)

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    # first find max values for numerical stability
    max_vals = buffer_new(ctx, (input.shape[0],))
    prg = clbuild(ctx.cl_ctx, """
    __kernel void max_vals(
        __global const float *a_g, int sz, __global float *res_g)
    {
      int gid = get_global_id(0);
      int gidsz = gid*sz;
      float max_val = -INFINITY;
      for (int x = 0; x < sz; x++) {
        max_val = max(max_val, a_g[gidsz+x]);
      }
      res_g[gid] = max_val;
    }
    """)
    prg.max_vals(ctx.cl_queue, [input.shape[0]], None, input, np.int32(input.shape[1]), max_vals)

    # compute exp(x - max) and sum
    lsum = buffer_new(ctx, (input.shape[0],))
    prg = clbuild(ctx.cl_ctx, """
    __kernel void logsoftmax(
        __global const float *a_g, __global const float *max_vals, int sz, __global float *res_g)
    {
      int gid = get_global_id(0);
      int gidsz = gid*sz;
      float max_val = max_vals[gid];
      float out = 0.0;
      for (int x = 0; x < sz; x++) {
        out += exp(a_g[gidsz+x] - max_val);
      }
      res_g[gid] = log(out) + max_val;
    }
    """)
    prg.logsoftmax(ctx.cl_queue, [input.shape[0]], None, input, max_vals, np.int32(input.shape[1]), lsum)

    # compute final output
    output = buffer_like(ctx, input)
    prg = clbuild(ctx.cl_ctx, """
    __kernel void lsmsub(
        __global const float *a_g, __global const float *b_g, int sz, __global float *res_g)
    {
      int gid = get_global_id(0);
      int gid2 = get_global_id(1);
      res_g[gid*sz + gid2] = a_g[gid*sz + gid2] - b_g[gid];
    }
    """)
    prg.lsmsub(ctx.cl_queue, [input.shape[0], input.shape[1]], None, input, lsum, np.int32(input.shape[1]), output)
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors
    grad_input = buffer_like(ctx, grad_output)
    prg = clbuild(ctx.cl_ctx, """
    __kernel void lsmsub2(
        __global const float *grad_output, __global const float *output, int sz, __global float *grad_input)
    {
      int gid = get_global_id(0);
      int gidsz = gid*sz;
      int gid2 = get_global_id(1);
      float acc = 0.0;
      for (int x = 0; x < sz; x++) {
        acc += grad_output[gidsz + x];
      }
      grad_input[gidsz + gid2] = grad_output[gidsz + gid2] - exp(output[gidsz + gid2]) * acc;
    }
    """)
    prg.lsmsub2(ctx.cl_queue, [grad_output.shape[0], grad_output.shape[1]], None, grad_output, output, np.int32(grad_output.shape[1]), grad_input)
    return grad_input
register('logsoftmax', LogSoftmax, gpu=True)

class Sigmoid(Function):
  @staticmethod
  def forward(ctx, input):
    ret = unary_op(ctx, '1./(1+exp(-a))', input)
    ctx.save_for_backward(ret)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    ret, = ctx.saved_tensors
    return binary_op(ctx, 'a * (b * (1 - b));', grad_output, ret)
register('sigmoid', Sigmoid, gpu=True)

# ***** CONV OPS ********

class Conv2D(Function):
  @staticmethod
  def forward(ctx, x, w, stride=1, groups=1):
    if type(ctx.stride) == int: # ctx stores function params
      ctx.stride = (ctx.stride, ctx.stride)
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_,iy,ix = x.shape
    oy,ox = (iy-(H-ys))//ys, (ix-(W-xs))//xs
    assert cin*ctx.groups == cin_
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups
    # output buffer
    ret = buffer_new(ctx, (bs, cout, oy, ox))
    prg = clbuild(ctx.cl_ctx, """
    __kernel void conv(__global const float *input, __global const float *weight, __global float *output,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs) {
      int B = get_global_id(0)/(groups*rcout);  // range 0-bs
      int g = (get_global_id(0)/rcout)%groups;
      int c = get_global_id(0) % rcout;
      int Y = get_global_id(1);  // range 0-oy
      int X = get_global_id(2);  // range 0-ox
      int IY = Y*ys;
      int IX = X*xs;
      
      // input  = (bs, groups, cin, iy, ix)
      // weight = (groups, rcout, cin, H, W)
      // output = (bs, groups, rcout, oy, ox)
      float acc = 0.0;
      for (int ci = 0; ci < cin; ci++) {
        for (int y = IY; y < IY+H; y++) {
          for (int x = IX; x < IX+W; x++) {
            acc += input[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + y*ix + x] * \
              weight[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + (y-IY)*W + (x-IX)];
          }
        }
      }
      output[B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X] = acc;
    }
    """)

    prg.conv(ctx.cl_queue, [bs*groups*rcout, oy, ox], None,
      x, w, ret,
      np.int32(H), np.int32(W),
      np.int32(groups), np.int32(rcout), np.int32(cin),
      np.int32(oy), np.int32(ox), 
      np.int32(iy), np.int32(ix),
      np.int32(ys), np.int32(xs)
    )
    return ret
  @staticmethod
  def backward(ctx, grad_output):
    raise Exception("not implemented")

register('conv2d', Conv2D, gpu=True)

class Pad2D(Function):
  @staticmethod
  def forward(ctx, x, padding=None):
    bs,cin,iy,ix = x.shape
    oy,ox = iy+padding[0]+padding[1], ix+padding[2]+padding[3] # top, bottom, left, right
    ret = buffer_zeros(ctx, (bs, cin, oy, ox))

    prg = clbuild(ctx.cl_ctx, """
    __kernel void pad2d(
        __global const float *input, __global float *output,
        int cin, int py, int px, int oy, int ox, int iy, int ix
      )
    {
      int B = get_global_id(0);
      int C = get_global_id(1);
      int Y = get_global_id(2);
      int iptr = B*cin*iy*ix + C*iy*ix + Y*ix;
      int optr = B*cin*oy*ox + C*oy*ox + (Y+py)*ox + px;
      for (int x = 0; x < ix; x++) {
        output[optr+x] = input[iptr+x];
      }
    }
    """)
    prg.pad2d(ctx.cl_queue, [bs, cin, iy], None,
        x, ret,
        np.int32(cin), np.int32(padding[0]), np.int32(padding[2]),
        np.int32(oy), np.int32(ox), np.int32(iy), np.int32(ix)
      )
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    raise Exception("write this")
register('pad2d', Pad2D, gpu=True)

class AvgPool2D(Function):
  @staticmethod
  def forward(ctx, input, kernel_size=(2, 2)):
    iter_op = "group_res += input[iid]"
    result_op = "group_res / (kernel_size.x * kernel_size.y)"
    return pooling_op(ctx, input, kernel_size, iter_op, result_op)

  @staticmethod
  def backward(ctx, grad_output):
    # for average pooling, we need to distribute the gradient evenly across all elements in the pooling window
    input_shape = ctx.data.shape
    N, C, Y, X = input_shape
    py, px = ctx.kernel_size
    ret = buffer_zeros(ctx, input_shape)
    
    prg = clbuild(ctx.cl_ctx, """
    __kernel void avgpool_backward(
      __global float *grad_input, __global const float *grad_output,
      uint2 osize, uint2 isize, uint2 kernel_size, int nelem
    ) {
      int3 gid = (int3)(get_global_id(2), get_global_id(1), get_global_id(0));
      int oid = gid.x + osize.x*(gid.y + osize.y*gid.z);
      float grad = grad_output[oid] / (kernel_size.x * kernel_size.y);
      
      for (uint j=0; j<kernel_size.y; ++j) {
        for (uint i=0; i<kernel_size.x; ++i) {
          int iid = (gid.x*kernel_size.x+i) + isize.x*((gid.y*kernel_size.y+j) + isize.y*gid.z);
          if (iid < nelem)
            grad_input[iid] += grad;
        }
      }
    }
    """)
    
    osize = np.array((X//px, Y//py), dtype=cl.cltypes.uint2)
    isize = np.array((X, Y), dtype=cl.cltypes.uint2)
    ksize = np.array((px,py), dtype=cl.cltypes.uint2)
    prg.avgpool_backward(ctx.cl_queue, (N*C, Y//py, X//px), None, ret, grad_output, osize, isize, ksize, np.int32(input_shape.size))
    return ret
register('avg_pool2d', AvgPool2D, gpu=True)

class MaxPool2D(Function):
  @staticmethod
  def forward(ctx, input, kernel_size=(2, 2)):
    init_val = "FLT_MIN"
    iter_op = "group_res = max(group_res, input[iid])"
    result_op = "group_res"
    ret = pooling_op(ctx, input, kernel_size, iter_op, result_op, init_val=init_val)
    
    # save indices of max elements for backward pass
    indices = buffer_new(ctx, ret.shape)
    prg = clbuild(ctx.cl_ctx, """
    __kernel void maxpool_indices(
      __global const float *input, __global float *output, __global int *indices,
      uint2 osize, uint2 isize, uint2 kernel_size, int nelem
    ) {
      int3 gid = (int3)(get_global_id(2), get_global_id(1), get_global_id(0));
      int oid = gid.x + osize.x*(gid.y + osize.y*gid.z);
      float max_val = -INFINITY;
      int max_idx = 0;
      
      for (uint j=0; j<kernel_size.y; ++j) {
        for (uint i=0; i<kernel_size.x; ++i) {
          int iid = (gid.x*kernel_size.x+i) + isize.x*((gid.y*kernel_size.y+j) + isize.y*gid.z);
          if (iid < nelem) {
            float val = input[iid];
            if (val > max_val) {
              max_val = val;
              max_idx = iid;
            }
          }
        }
      }
      indices[oid] = max_idx;
    }
    """)
    
    N, C, Y, X = input.shape
    py, px = kernel_size
    osize = np.array((X//px, Y//py), dtype=cl.cltypes.uint2)
    isize = np.array((X, Y), dtype=cl.cltypes.uint2)
    ksize = np.array((px,py), dtype=cl.cltypes.uint2)
    prg.maxpool_indices(ctx.cl_queue, (N*C, Y//py, X//px), None, input, ret, indices, osize, isize, ksize, np.int32(input.size))
    ctx.save_for_backward(indices)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    indices, = ctx.saved_tensors
    input_shape = ctx.data.shape
    ret = buffer_zeros(ctx, input_shape)
    prg = clbuild(ctx.cl_ctx, """
    __kernel void maxpool_backward(
      __global float *grad_input, __global const float *grad_output,
      __global const int *indices, int nelem
    ) {
      int gid = get_global_id(0);
      if (gid < nelem) {
        int idx = indices[gid];
        grad_input[idx] += grad_output[gid];
      }
    }
    """)
    prg.maxpool_backward(ctx.cl_queue, [np.prod(grad_output.shape)], None, ret, grad_output, indices, np.int32(grad_output.size))
    return ret
register('max_pool2d', MaxPool2D, gpu=True)

class Dropout(Function):
  @staticmethod
  def forward(ctx, input, p=0.5, training=True):
    if not training: return input
    else: raise NotImplementedError("GPU dropout in training mode is not yet implemented.")

  @staticmethod
  def backward(ctx, grad_output):
    if not ctx.training: return grad_output
    else: raise NotImplementedError("GPU dropout backward in training mode is not yet implemented.")

register("dropout", Dropout, gpu=True)