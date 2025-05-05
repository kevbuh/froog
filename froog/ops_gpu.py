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
from .tensor import Function, register
import pyopencl as cl
import functools

def buffer_new(ctx, shape):
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, 4*np.prod(shape))
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_zeros(ctx, shape):
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.zeros(shape))
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_like(ctx, x):
  return buffer_new(ctx, x.shape)

@functools.lru_cache
def clbuild(cl_ctx, prg):
  return cl.Program(cl_ctx, prg).build()

def binary_op(ctx, code, x, y):
  # if len(x.shape) != len(y.shape):
  #   raise Exception(f"shape mismatch in binop {code}: {x.shape} {y.shape}")
  xdiv = 1
  ydiv = 1
  if x.shape != y.shape:
    # special case broadcasting
    if len(y.shape) == 4 and x.shape[0:2] == y.shape[0:2] and y.shape[2] == 1 and y.shape[3] == 1:
      ydiv = x.shape[2] * x.shape[3]
    elif len(y.shape) == 4 and x.shape[0:2] == y.shape[0:2] and x.shape[2] == 1 and x.shape[3] == 1:
      xdiv = y.shape[2] * y.shape[3]
    elif np.prod(y.shape) == 1:
      ydiv = np.prod(x.shape)
    else:
      raise Exception(f"binary op shape mismatch: {x.shape} != {y.shape}")
  ret = buffer_like(ctx, x if np.prod(x.shape) >= np.prod(y.shape) else y)
  prg = clbuild(ctx.cl_ctx, """
  __kernel void binop(  __global const float *a_g, __global const float *b_g, __global float *res_g, int xdiv, int ydiv) {
    int gid = get_global_id(0);
    float a = a_g[gid/xdiv];
    float b = b_g[gid/ydiv];
    res_g[gid] = """+code+""";
  }
  """)
  prg.binop(ctx.cl_queue, [np.prod(ret.shape)], None, x, y, ret, np.int32(xdiv), np.int32(ydiv))
  return ret

def unary_op(ctx, code, x):
  ret = buffer_like(ctx, x)
  prg = clbuild(ctx.cl_ctx, """
  __kernel void unop(
      __global const float *a_g, __global float *res_g)
  {
    int gid = get_global_id(0);
    float a = a_g[gid];
    res_g[gid] = """+code+""";
  }
  """)
  prg.unop(ctx.cl_queue, [np.prod(ret.shape)], None, x, ret)
  return ret

@functools.lru_cache
def cl_pooling_krnl_build(cl_ctx, iter_op, result_op, init_val=0):
  prg = """
  __kernel void subsample(
    __global float *output, __global const float *input, uint2 osize, uint2 isize, uint2 kernel_size, int nelem
  ) {
    int3 gid = (int3)(get_global_id(2), get_global_id(1), get_global_id(0));
    int oid = gid.x + osize.x*(gid.y + osize.y*gid.z);
    float group_res = """+str(init_val)+""";
    for (uint j=0; j<kernel_size.y; ++j) {
      for (uint i=0; i<kernel_size.x; ++i) {
        int iid  = (gid.x*kernel_size.x+i) + isize.x*((gid.y*kernel_size.y+j) + isize.y*gid.z);
        if (iid < nelem)
          """+iter_op+""";
      }
    }
    output[oid] = """+result_op+""";
  }
  """
  return clbuild(cl_ctx, prg)

def pooling_op(ctx, input, kernel_size, iter_op, result_op, init_val=0):
  N, C, Y, X = input.shape
  py,px = kernel_size
  ret = buffer_new(ctx, (N, C, Y//py, X//px))
  osize = np.array((X//px, Y//py), dtype=cl.cltypes.uint2)
  isize = np.array((X, Y), dtype=cl.cltypes.uint2)
  ksize = np.array((px,py), dtype=cl.cltypes.uint2)
  prg = cl_pooling_krnl_build(ctx.cl_ctx, iter_op, result_op, init_val=init_val)
  prg.subsample(ctx.cl_queue, (N*C, Y//py, X//px), None, ret, input, osize, isize, ksize, np.int32(input.size))
  ctx.data = np.empty((N, C, Y, X)) # set shape expectation on tensor instance
  return ret

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
    gradx = binary_op(ctx, 'a*b', grad_output,
                      binary_op(ctx, 'b * (pow((float)a, (float)(b-1.0)));', x, y))
    grady = binary_op(ctx, 'a*b', grad_output,
                      binary_op(ctx, 'pow((float)a, (float)b) * log(a);', x, y))
    return gradx, grady
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
    prg.sum(ctx.cl_queue, [input.shape[0]], None, input, np.int32(np.prod(input.shape)), ret)
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
    prg.fill(ctx.cl_queue, [np.prod(ret.shape)], None, grad_output, ret)
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
    ss = list(shape)

    # ???
    tsum = 1
    for s in ss:
      if s != -1:
        tsum *= s
    for i,s in enumerate(ss):
      if s == -1:
        ss[i] = np.prod(x.shape) // tsum
    assert np.prod(x.shape) == np.prod(ss)
    x.shape = tuple(ss)
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
      // TODO: this is repeated in many kernels
      float acc = 0.0;
      for (int x = 0; x < sz; x++) {
        acc += grad_output[gidsz + x];
      }
      grad_input[gidsz + gid2] = grad_output[gidsz + gid2] - exp(output[gidsz + gid2]) * acc;
    }
    """)
    prg.lsmsub2(ctx.cl_queue, [grad_output.shape[0], grad_output.shape[1]], None,
      grad_output, output, np.int32(grad_output.shape[1]), grad_input)

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