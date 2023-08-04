#  _______  ______    _______  _______  _______ 
# |       ||    _ |  |       ||       ||       |
# |    ___||   | ||  |   _   ||   _   ||    ___|
# |   |___ |   |_||_ |  | |  ||  | |  ||   | __ 
# |    ___||    __  ||  |_|  ||  |_|  ||   ||  |
# |   |    |   |  | ||       ||       ||   |_| |
# |___|    |___|  |_||_______||_______||_______|

import numpy as np
from .tensor import Function, register, Tensor
import pyopencl as cl
import functools
import pyopencl.array as pycl_array
from pyopencl.reduction import ReductionKernel

def buffer_new(ctx, shape):
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, 4*np.prod(shape))
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_like(ctx, x):
  return buffer_new(ctx, x.shape)

@functools.lru_cache
def clbuild(cl_ctx, prg):
  return cl.Program(cl_ctx, prg).build()

def binary_op(ctx, code, x, y): 
  ret = buffer_like(ctx, x)
  prg = clbuild(ctx.cl_ctx, """
  __kernel void add(
      __global const float *a_g, __global const float *b_g, __global float *res_g)
  {
    int gid = get_global_id(0);
    """+code+"""
  }
  """)
  prg.add(ctx.cl_queue, [np.prod(ret.shape)], None, x, y, ret) # (queue, size, ???, arg1, arg2, dest)
  return ret

def unary_op(ctx, code, x):
  ret = buffer_like(ctx, x)
  prg = clbuild(ctx.cl_ctx, """
  __kernel void relu(
      __global const float *a_g, __global float *res_g)
  {
    int gid = get_global_id(0);
    """+code+"""
  }
  """)
  prg.relu(ctx.cl_queue, [np.prod(ret.shape)], None, x, ret)
  return ret

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    return binary_op(ctx, 'res_g[gid] = a_g[gid] + b_g[gid];', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add, gpu=True)

class Sub(Function):
  @staticmethod
  def forward(ctx, x, y):
    return binary_op(ctx, 'res_g[gid] = a_g[gid] - b_g[gid];', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    not_grad_output = unary_op(ctx, 'res_g[gid] = -a_g[gid];', grad_output)
    return grad_output, not_grad_output
register('sub', Sub, gpu=True)

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)

    # HACK
    if y.shape == (1,):
      return binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[0];', x, y)
    elif x.shape == y.shape:
      return binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[gid];', x, y)
    else:
      raise Exception("mismatched shapes %r %r" % (x.shape, y.shape))

    return ret

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[gid];', y, grad_output),\
           binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[gid];', x, grad_output)
register('mul', Mul, gpu=True)

class Pow(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(ctx, 'res_g[gid] = pow(a_g[gid], b_g[gid]);', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    gradx = binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[gid];', grad_output,
                      binary_op(ctx, 'res_g[gid] = b_g[gid] * (pow((float)a_g[gid], (float)(b_g[gid]-1.0)));', x, y))
    grady = binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[gid];', grad_output,
                      binary_op(ctx, 'res_g[gid] = pow((float)a_g[gid], (float)b_g[gid]) * log(a_g[gid]);', x, y))
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
    return binary_op(ctx, 'res_g[gid] = b_g[0];', input, grad_output)  # Quick hack for fill
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

class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op(ctx, 'res_g[gid] = max(a_g[gid], (float)0.);', input)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(ctx, 'res_g[gid] = a_g[gid] * (b_g[gid] >= 0);', grad_output, input)
register('relu', ReLU, gpu=True)

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    lsum = buffer_new(ctx, (input.shape[0],))
    prg = clbuild(ctx.cl_ctx, """
    __kernel void logsoftmax(
        __global const float *a_g, int sz, __global float *res_g)
    {
      int gid = get_global_id(0);
      int gidsz = gid*sz;
      // TODO: stability with max
      float out = 0.0;
      for (int x = 0; x < sz; x++) {
        out += exp(a_g[gidsz+x]);
      }
      res_g[gid] = log(out);
    }
    """)
    prg.logsoftmax(ctx.cl_queue, [input.shape[0]], None, input, np.int32(input.shape[1]), lsum)

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

