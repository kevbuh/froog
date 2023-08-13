#  _______  ______    _______  _______  _______ 
# |       ||    _ |  |       ||       ||       |
# |    ___||   | ||  |   _   ||   _   ||    ___|
# |   |___ |   |_||_ |  | |  ||  | |  ||   | __ 
# |    ___||    __  ||  |_|  ||  |_|  ||   ||  |
# |   |    |   |  | ||       ||       ||   |_| |
# |___|    |___|  |_||_______||_______||_______|
#
# inspired by pytorch
# inspired by tinygrad
# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py

from functools import partialmethod
from inspect import signature
import numpy as np
import os

try:
  import pyopencl as cl
  GPU = True
except ImportError:
  # no GPU support
  GPU = False

cl_ctx, cl_queue = None, None
def init_gpu():
  """
  creates global OpenCL context and queue
  """
  global cl_ctx, cl_queue
  if cl_queue is None:
    try:
      # if you have an m2 mac 
      cl_ctx = cl.create_some_context(answers=[0]) 
    except (cl._cl.RuntimeError, TypeError):
      cl_ctx = cl.create_some_context(interactive=False)
    cl_queue = cl.CommandQueue(cl_ctx)

# ************ Main Classes ************
# ********** Tensor, Function **********
#   _____________   _______ ____  ____ 
#  /_  __/ ____/ | / / ___// __ \/ __ \
#   / / / __/ /  |/ /\__ \/ / / / /_/ /
#  / / / /___/ /|  /___/ / /_/ / _, _/ 
# /_/ /_____/_/ |_//____/\____/_/ |_|  

class Tensor:
  did_float_warning = False
  def __init__(self, data, gpu=False):
    if isinstance(data, list):
      data = np.array(data, dtype=np.float32)
    elif GPU and isinstance(data, cl._cl.Buffer):
      self.gpu = True
    elif not isinstance(data, np.ndarray):
      raise TypeError(f"Error constructing tensor with {data}")
    
    if isinstance(data, np.ndarray):
      if data.dtype != np.float32 and not Tensor.did_float_warning:
        # TODO: set env flag to print all warnings, float64 needed for numerical jacobian
        print(f"warning, {data.shape} isn't float32")
        if not os.getenv("DEBUG") == "1":
          Tensor.did_float_warning = True
      self.gpu = False

    self.data = data
    self.grad = None # TODO: why self.grad.data instead of self.grad?

    if gpu:
      self.gpu_()

    # internal variables used for autograd graph construction
    self._ctx = None # these are where the backward gradient computation are saved

  def __repr__(self):
    return f"Tensor data: {self.data}, gradients: {self.grad.data if self.grad else None}" 
  
  def assign(self, x):
    self.data = x.data

  @property
  def shape(self):
    return self.data.shape
  
  @staticmethod
  def zeros(*shape):
    return Tensor(np.zeros(shape, dtype=np.float32))
  
  @staticmethod
  def ones(*shape):
    return Tensor(np.ones(shape, dtype=np.float32))
  
  @staticmethod
  def randn(*shape):
    return Tensor(np.random.randn(*shape).astype(np.float32))
  
  @staticmethod
  def eye(dim):
    return Tensor(np.eye(dim).astype(np.float32))

  def backward(self, allow_fill=True): 
    if self._ctx is None:
      return

    if self.grad is None and allow_fill:
      # allow_fill gives backprop a starting point, fills in the first grad with one is its None
      assert self.data.shape == (1,) # numpy returns tuples as shapes
      self.grad = Tensor(np.ones(self.data.shape, dtype=self.data.dtype), gpu=self.gpu)

    assert self.grad is not None
    
    # THIS IS WHERE AUTO GRAD IS DONE
    grads = self._ctx.backward(self._ctx, self.grad.data) # get gradients respective to what op happened
    if len(self._ctx.parents) == 1:
      grads = [grads]
    for t, g in zip(self._ctx.parents, grads):
      if g is None:
        continue
      if g.shape != t.data.shape:
        print(f"grad shape must match tensor shape in {self._ctx}, {g.shape} != {t.data.shape}")
        assert False
      t.grad = Tensor(g) # access actual gradients using grad.data
      t.backward(allow_fill=False) 

  # ****** cpu/gpu ******
    
  def to_cpu(self):
    if self.gpu:
      data = np.empty(self.shape, dtype=np.float32)
      cl.enqueue_copy(cl_queue, data, self.data) # copy data from cpu to gpu (queue, dest, src)
      ret = Tensor(data)
      if self.grad:
        ret.grad = self.grad.to_cpu()
      return ret
    else: 
      return self
    
  def gpu_(self):
    self.data = self.to_gpu().data
    self.gpu = True
  
  def to_gpu(self):
    if not GPU:
      raise Exception("no gpu support! install pyopencl")
    if not self.gpu:
      init_gpu()
      assert self.data.dtype == np.float32 # GPU only allows float32
      # hostbuf is the data buffer on host machine with the data to be copied to the OpenCL buffer
      data = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data.ravel()) # from pyopencl docs
      data.shape = self.shape
      data.dtype = self.data.dtype
      ret = Tensor(data)
      if self.grad:
        ret.grad = self.grad.to_gpu()
      return ret
    else:
      return self 

  ops = {}     # stores operations that are done on the CPU
  ops_gpu = {} # stores operations that are done on the GPU

  # ****** basic tensor math ops ******

  def mean(self):
    div = Tensor(np.array([1 / np.prod(self.shape)], dtype=self.data.dtype), gpu=self.gpu)
    return self.sum().mul(div)
  
  def sqrt(self):
    root = Tensor(np.zeros(self.shape, dtype=self.data.dtype)+0.5, gpu=self.gpu)
    return self.pow(root)

  def div(self, y):
    root = Tensor(np.zeros(self.shape, dtype=self.data.dtype)-1, gpu=self.gpu)
    return self.mul(y.pow(root))

#     ________  ___   ______________________  _   __
#    / ____/ / / / | / / ____/_  __/  _/ __ \/ | / /
#   / /_  / / / /  |/ / /     / /  / // / / /  |/ / 
#  / __/ / /_/ / /|  / /___  / / _/ // /_/ / /|  /  
# /_/    \____/_/ |_/\____/ /_/ /___/\____/_/ |_/     
                    
class Function:
  """
  An instantiation of the Function class includes the context
  """
  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  def apply(self, *x, **kwargs):
    """
    self  : is the tensor with data
    *x    : the input to the method  
    """
    op = self # self is the operation class
    ctx = op(*x)
    params = signature(op.forward).parameters # gets the function params e.g. (ctx, x, y)
    for p in params.values():                 # loops through each param
      if p.default is not p.empty:            # p.default is the param value
        setattr(ctx, p.name, p.default)       # add any func params to ctx
    for k, v in kwargs.items(): 
      setattr(ctx, k, v)                      # add any kwargs to ctx

    # this performs the actual operation (e.g., addition, multiplication, etc.) on the tensor data
    ret = Tensor(op.forward(ctx, *[t.data for t in x], **kwargs))
    ret._ctx = ctx
    return ret

def register(name, fxn, gpu=False):
  """
  mechanism that allows you to chain methods in an intuitive and Pythonic way
  e.g. x.dot(w).relu(), where w is a tensor

  partialmethod is used to create a new method that has some of the arguments to
  another method already filled in the apply method of that instance is added
  """
  if gpu: 
    Tensor.ops_gpu[name] = fxn
  else:
    Tensor.ops[name] = fxn

  def dispatch(self, *x, **kwargs):
    op_func = (Tensor.ops_gpu if self.gpu else Tensor.ops)[name]
    op_func.cl_ctx, op_func.cl_queue = cl_ctx, cl_queue
    return op_func.apply(op_func, self, *x, **kwargs)
  
  setattr(Tensor, name, dispatch)

  if name in ['add', 'sub', 'mul', 'div']:
    setattr(Tensor, "__%s__" % name, dispatch)
    setattr(Tensor, "__i%s__" % name, lambda self,x: self.assign(dispatch(self,x)))

import froog.ops # this registers all the operations
if GPU:
  import froog.ops_gpu