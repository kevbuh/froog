#  _______  ______    _______  _______  _______ 
# |       ||    _ |  |       ||       ||       |
# |    ___||   | ||  |   _   ||   _   ||    ___|
# |   |___ |   |_||_ |  | |  ||  | |  ||   | __ 
# |    ___||    __  ||  |_|  ||  |_|  ||   ||  |
# |   |    |   |  | ||       ||       ||   |_| |
# |___|    |___|  |_||_______||_______||_______|

import numpy as np
import pyopencl as cl
from froog.tensor import Function, register, Tensor

def buffer_new(ctx, shape):
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, 4*np.prod(shape)) # size of the buffer in bytes
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_like(ctx, x):
  return buffer_new(ctx, x.shape)

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    ret = buffer_like(ctx, x)
    prg = cl.Program(ctx.cl_ctx, """
    __kernel void add( __global const float *a_g, __global const float *b_g, __global float *res_g) {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """).build()
    prg.add(ctx.cl_queue, [ret.size//4], None, x, y, ret)
    return ret
  
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add, gpu=True)

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ret = buffer_like(ctx, x)
    prg = cl.Program(ctx.cl_ctx, """
    __kernel void mul( __global const float *a_g, __global const float *b_g, __global float *res_g) {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] * b_g[gid];
    }
    """).build()
    prg.mul(ctx.cl_queue, [ret.size//4], None, x, y, ret)
    ctx.save_for_backward(x, y, prg)
    return ret
  
  @staticmethod
  def backward(ctx, grad_output):
    x, y, prg = ctx.saved_tensors
    gx = buffer_like(ctx, x) # allocate buffer
    gy = buffer_like(ctx, y) # allocate buffer
    prg.mul(ctx.cl_queue, [gx.size//4], None, y, grad_output, gx)  # (queue, ???, ???, arg1, arg2, dest)
    prg.mul(ctx.cl_queue, [gy.size//4], None, x, grad_output, gy)
    return gx, gy
register('mul', Mul, gpu=True)

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    ret = buffer_new(ctx, (1,)) # buffer of size 1, which will hold the sum
    prg = cl.Program(ctx.cl_ctx, """
    __kernel void sum(
        __global const float *a_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[0] += a_g[gid];
    }
    """).build()
    prg.sum(ctx.cl_queue, [input.size//4], None, input, ret)
    return ret
  
  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    ret = Tensor(grad_output).to_cpu().data * np.ones(input.shape, dtype=input.dtype) # gradient * ddx(x)
    return Tensor(ret).to_gpu().data
register('sum', Sum, gpu=True)

class Dot(Function):
  @staticmethod
  def forward(ctx, x, w):
    """
    A[gid_y * size + i] accesses an element in the row gid_y of matrix A and the column i
    """
    assert x.shape[1] == w.shape[0] # inner dims must match for dot product
    input_size = np.int32(x.shape[0])
    inner_size = np.int32(x.shape[1]) 
    outer_size = np.int32(w.shape[1])
    one = np.int32(1)
    ret = buffer_new(ctx, (input_size, outer_size)) # TODO: why not cl_ctx?

    prg = cl.Program(ctx.cl_ctx, """
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
    """).build()

    ctx.save_for_backward(x, w, prg)
    prg.matmul(ctx.cl_queue, [input_size, outer_size], None, x, w, ret, inner_size, one, inner_size, one, outer_size, outer_size)
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
      one, msize, isize, one, isize, osize)

    return grad_input, grad_weight
register('dot', Dot, gpu=True)
register('matmul', Dot, gpu=True)