import numpy as np

# check if OpenCL is available
cl_ctx, cl_queue = None, None
try:
  import pyopencl as cl
  GPU = True
except ImportError:
  cl = None
  GPU = False

def init_gpu():
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

def tensor_to_cpu(tensor):
  """Convert a GPU tensor to CPU"""
  if tensor.gpu:
    data = np.empty(tensor.shape, dtype=np.float32)
    cl.enqueue_copy(cl_queue, data, tensor.data)
    return data
  else:
    return tensor.data

def tensor_to_gpu(data):
  """Convert CPU data to GPU buffer"""
  if not GPU: raise Exception("no gpu support! install pyopencl")
  init_gpu()
  assert data.dtype == np.float32 # GPU only allows float32
  gpu_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data.ravel())
  gpu_buffer.shape = data.shape
  gpu_buffer.dtype = data.dtype
  return gpu_buffer

def is_buffer(data):
  """Check if data is a GPU buffer"""
  return GPU and isinstance(data, cl._cl.Buffer) 