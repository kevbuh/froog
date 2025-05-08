import numpy as np
from typing import Any, Tuple, Optional, Union, TypeVar, cast, List, Dict, Callable
import functools

# check if OpenCL is available
cl_ctx, cl_queue = None, None
try:
  import pyopencl as cl
  GPU = True
except ImportError:
  cl = None
  GPU = False

def init_gpu() -> None:
  """
  creates global OpenCL context and queue
  """
  global cl_ctx, cl_queue
  if cl_queue is None and GPU:
    try:
      cl_ctx = cl.create_some_context(answers=[0]) 
    except (cl._cl.RuntimeError, TypeError):
      cl_ctx = cl.create_some_context(interactive=False)
    cl_queue = cl.CommandQueue(cl_ctx)

def tensor_to_cpu(tensor: Any) -> np.ndarray:
  """Convert a GPU tensor to CPU"""
  if tensor.gpu:
    data = np.empty(tensor.shape, dtype=np.float32)
    cl.enqueue_copy(cl_queue, data, tensor.data)
    return data
  else:
    return tensor.data

def tensor_to_gpu(data: np.ndarray) -> Any:
  """Convert CPU data to GPU buffer"""
  if not GPU: raise Exception("no gpu support! install pyopencl")
  init_gpu()
  assert data.dtype == np.float32 # GPU only allows float32
  gpu_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data.ravel())
  gpu_buffer.shape = data.shape
  gpu_buffer.dtype = data.dtype
  return gpu_buffer

def is_buffer(data: Any) -> bool:
  """Check if data is a GPU buffer"""
  return GPU and isinstance(data, cl._cl.Buffer)

# Helper functions moved from ops_gpu.py

def get_size(x: Any) -> int:
  """Return the total number of elements in x"""
  return int(np.prod(x.shape))

def buffer_new(ctx: Any, shape: Tuple[int, ...]) -> Any:
  """Create a new empty GPU buffer with the given shape"""
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, 4*get_size(shape))
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_zeros(ctx: Any, shape: Tuple[int, ...]) -> Any:
  """Create a new GPU buffer filled with zeros"""
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.zeros(shape))
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_like(ctx: Any, x: Any) -> Any:
  """Create a new GPU buffer with the same shape as x"""
  return buffer_new(ctx, x.shape)

@functools.lru_cache
def clbuild(cl_ctx: Any, prg: str) -> Any:
  """Build an OpenCL program"""
  if GPU:
    return cl.Program(cl_ctx, prg).build()
  return None

def binary_op(ctx: Any, code: str, x: Any, y: Any) -> Any:
  """Apply a binary operation to two GPU tensors"""
  xdiv = 1
  ydiv = 1
  if x.shape != y.shape:
    # special case broadcasting
    if len(y.shape) == 4 and x.shape[0:2] == y.shape[0:2] and y.shape[2] == 1 and y.shape[3] == 1:
      ydiv = x.shape[2] * x.shape[3]
    elif len(y.shape) == 4 and x.shape[0:2] == y.shape[0:2] and x.shape[2] == 1 and x.shape[3] == 1:
      xdiv = y.shape[2] * y.shape[3]
    elif get_size(y) == 1:
      ydiv = get_size(x)
    else:
      raise Exception(f"binary op shape mismatch: {x.shape} != {y.shape}")
  ret = buffer_like(ctx, x if get_size(x) >= get_size(y) else y)
  prg = clbuild(ctx.cl_ctx, """
  __kernel void binop(  __global const float *a_g, __global const float *b_g, __global float *res_g, int xdiv, int ydiv) {
    int gid = get_global_id(0);
    float a = a_g[gid/xdiv];
    float b = b_g[gid/ydiv];
    res_g[gid] = """+code+""";
  }
  """)
  prg.binop(ctx.cl_queue, [get_size(ret)], None, x, y, ret, np.int32(xdiv), np.int32(ydiv))
  return ret

def unary_op(ctx: Any, code: str, x: Any) -> Any:
  """Apply a unary operation to a GPU tensor"""
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
  prg.unop(ctx.cl_queue, [get_size(ret)], None, x, ret)
  return ret

@functools.lru_cache
def cl_pooling_krnl_build(cl_ctx: Any, iter_op: str, result_op: str, init_val: Union[int, str] = 0) -> Any:
  """Build an OpenCL kernel for pooling operations"""
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

def pooling_op(ctx: Any, input: Any, kernel_size: Tuple[int, int], iter_op: str, result_op: str, init_val: Union[int, str] = 0) -> Any:
  """Apply a pooling operation to a GPU tensor"""
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