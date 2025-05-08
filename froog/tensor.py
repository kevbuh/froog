#  _______  ______    _______  _______  _______ 
# |       ||    _ |  |       ||       ||       |
# |    ___||   | ||  |   _   ||   _   ||    ___|
# |   |___ |   |_||_ |  | |  ||  | |  ||   | __ 
# |    ___||    __  ||  |_|  ||  |_|  ||   ||  |
# |   |    |   |  | ||       ||       ||   |_| |
# |___|    |___|  |_||_______||_______||_______|

import os
import numpy as np
from inspect import signature
from typing import Tuple, List, Union, Optional, Any, TypeVar, cast

# Import device utilities
from froog.gpu import get_device, set_device
from froog.gpu.device import Device

# Check if any GPU is available
DEVICE = get_device()
GPU = DEVICE is not None

T = TypeVar('T', bound='Tensor') # For self-referential types

# ************ Main Classes ************
# ********** Tensor, UOP **********
#   _____________   _______ ____  ____ 
#  /_  __/ ____/ | / / ___// __ \/ __ \
#   / / / __/ /  |/ /\__ \/ / / / /_/ /
#  / / / /___/ /|  /___/ / /_/ / _, _/ 
# /_/ /_____/_/ |_//____/\____/_/ |_|  

class Tensor:
  did_float_warning = False
  def __init__(self, data: Union[List, np.ndarray, Any], gpu: bool = False):
    if isinstance(data, list): data = np.array(data, dtype=np.float32)
    elif self.is_device_tensor(data): self.gpu = True
    elif not isinstance(data, np.ndarray): raise TypeError(f"Error constructing tensor with {data}")
    
    if isinstance(data, np.ndarray):
      if data.dtype != np.float32 and not Tensor.did_float_warning:
        # Only print warnings if WARNING env var is set to "1"
        if os.getenv("WARNING") == "1":
          print(f"warning, {data.shape} isn't float32. float64 needed for numerical jacobian")
        if not os.getenv("DEBUG") == "1":
          Tensor.did_float_warning = True
      self.gpu = False

    # internal variables used for autograd graph construction
    self.data = data
    self.grad: Optional[Tensor] = None
    self._ctx = None # these are where the backward gradient computation are saved
    if gpu: self.gpu_()

  def __repr__(self) -> str:
    return f"Tensor data: {self.data}, gradients: {self.grad.data if self.grad else None}" 
  
  def assign(self, x: T) -> None:
    self.data = x.data

  # ********** Properties **********

  @property
  def shape(self) -> Tuple[int, ...]:
    """
    The shape of the tensor as a tuple of dimensions.
    """
    return self.data.shape
  
  @property
  def size(self, dim=None) -> Union[int, Tuple[int, ...]]:
    """
    Total number of elements in the tensor or size in a specific dimension.
    """
    if dim is not None: return self.shape[dim]
    return int(np.prod(self.shape))
  
  @property
  def ndim(self) -> int:
    """
    Number of dimensions (rank) of the tensor.
    """
    return len(self.shape)
  
  @property
  def transpose(self) -> T:
    """
    Transpose of the tensor.
    """
    if isinstance(self.data, np.ndarray):
      return Tensor(self.data.T, gpu=self.gpu)
    else:
      # For GPU tensor, we need to bring it back to CPU, transpose, then back to GPU
      cpu_tensor = self.to_cpu()
      return Tensor(cpu_tensor.data.T, gpu=self.gpu)
  
  @property
  def dtype(self) -> np.dtype:
    """
    Data type of the tensor.
    """
    return self.data.dtype
  
  @property
  def is_gpu(self) -> bool:
    """
    True if the tensor is on GPU.
    """
    return self.gpu
  
  # ********** Methods **********
  
  @staticmethod
  def zeros(*shape: int) -> T:
    """
    Creates a tensor filled with zeros of the specified shape.
    """
    return Tensor(np.zeros(shape, dtype=np.float32))
  
  @staticmethod
  def ones(*shape: int) -> T:
    """
    Creates a tensor filled with ones of the specified shape.
    """
    return Tensor(np.ones(shape, dtype=np.float32))
  
  @staticmethod
  def randn(*shape: int) -> T:
    """
    Creates a tensor with random values from a normal distribution.
    """
    return Tensor(np.random.randn(*shape).astype(np.float32))
  
  @staticmethod
  def eye(dim: int) -> T:
    """
    Creates a 2D identity tensor of shape (dim, dim).
    """
    return Tensor(np.eye(dim).astype(np.float32))

  @staticmethod
  def arange(start: Union[int, float], stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1) -> T:
    """
    Creates a 1D tensor with evenly spaced values.
    If stop is None, start is interpreted as 0 and start becomes the stop value.
    """
    if stop is None:
      stop = start
      start = 0
    return Tensor(np.arange(start, stop, step, dtype=np.float32))
  
  def flatten(self) -> T:
    """
    Returns a flattened (1D) copy of the tensor.
    """
    return Tensor(self.data.reshape(-1), gpu=self.gpu)
  
  def detach(self) -> T:
    """
    Returns a new tensor detached from the current computation graph.
    """
    return Tensor(self.data.copy(), gpu=self.gpu)
  
  def view(self, *shape: int) -> T:
    """
    Returns a tensor with the same data but different shape.
    """
    return Tensor(self.data.reshape(shape), gpu=self.gpu)
  
  def unsqueeze(self, dim: int) -> T:
    """
    Adds a dimension of size 1 at the specified position.
    """
    shape = list(self.shape)
    if dim < 0:
        dim = len(shape) + 1 + dim
    shape.insert(dim, 1)
    return Tensor(self.data.reshape(shape), gpu=self.gpu)
  
  def squeeze(self, dim: Optional[int] = None) -> T:
    """
    Removes dimensions of size 1.
    If dim is specified, only removes the dimension at that position if it's 1.
    """
    if dim is None:
      return Tensor(self.data.squeeze(), gpu=self.gpu)
    else:
      shape = list(self.shape)
      if dim < 0:
        dim = len(shape) + dim
      if 0 <= dim < len(shape) and shape[dim] == 1:
        shape.pop(dim)
      return Tensor(self.data.reshape(shape), gpu=self.gpu)
  
  def to_float(self) -> T:
    """
    Converts tensor to float32 data type.
    """
    return Tensor(self.data.astype(np.float32), gpu=self.gpu)
  
  def to_int(self) -> T:
    """
    Converts tensor to int32 data type.
    """
    return Tensor(self.data.astype(np.int32), gpu=self.gpu)
  
  def to_bool(self) -> T:
    """
    Converts tensor to boolean data type.
    """
    return Tensor(self.data.astype(bool), gpu=self.gpu)

  # ********** Backward **********

  def backward(self, allow_fill: bool = True) -> None: 
    if self._ctx is None: return
    if self.grad is None and allow_fill:
      # allow_fill gives backprop a starting point, fills in the first grad with one is its None
      assert self.data.shape == (1,) # numpy returns tuples as shapes
      self.grad = Tensor(np.ones(self.data.shape, dtype=self.data.dtype), gpu=self.gpu)
    assert self.grad is not None
    # NOTE: THIS IS WHERE AUTO GRAD IS DONE
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
    
  @staticmethod
  def is_device_tensor(data: Any) -> bool:
    """Check if data is on a GPU device."""
    # Use device-specific checks if available
    if DEVICE:
      return DEVICE.is_device_tensor(data)
    return False
    
  def to_cpu(self) -> T:
    if self.gpu:
      if DEVICE:
        data = DEVICE.tensor_to_cpu(self.data)
        ret = Tensor(data)
        if self.grad:
          ret.grad = self.grad.to_cpu()
        return ret
      else:
        raise Exception("No GPU device available for tensor conversion")
    else: 
      return cast(T, self)
    
  def gpu_(self) -> None:
    """Move tensor to GPU in-place."""
    if not self.gpu and GPU:
      if DEVICE:
        self.data = DEVICE.tensor_to_device(self.data)
        self.gpu = True
        if self.grad:
          self.grad.gpu_()
      else:
        raise Exception("No GPU device available")
  
  def to_gpu(self) -> T:
    """Return a copy of this tensor on the GPU."""
    if not GPU:
      raise Exception("No GPU support available!")
    
    if not self.gpu:
      if DEVICE:
        gpu_data = DEVICE.tensor_to_device(self.data)
        ret = Tensor(gpu_data)
        ret.gpu = True
        if self.grad:
          ret.grad = self.grad.to_gpu()
        return ret
      else:
        raise Exception("No GPU device available")
    else:
      return cast(T, self)

  ops = {}     # stores operations that are done on the CPU
  ops_gpu = {} # stores operations that are done on the GPU

  # ****** basic tensor math ops ******

  def mean(self) -> T:
    div = Tensor(np.array([1 / self.size], dtype=self.data.dtype), gpu=self.gpu)
    return self.sum().mul(div)
  
  def sqrt(self) -> T:
    root = Tensor(np.zeros(self.shape, dtype=self.data.dtype)+0.5, gpu=self.gpu)
    return self.pow(root)

  def div(self, y: T) -> T:
    root = Tensor(np.zeros(self.shape, dtype=self.data.dtype)-1, gpu=self.gpu)
    return self.mul(y.pow(root))

#     ________  ___   ______________________  _   __
#    / ____/ / / / | / / ____/_  __/  _/ __ \/ | / /
#   / /_  / / / /  |/ / /     / /  / // / / /  |/ / 
#  / __/ / /_/ / /|  / /___  / / _/ // /_/ / /|  /  
# /_/    \____/_/ |_/\____/ /_/ /___/\____/_/ |_/     
                    
class UOP:
  """
  Base class for all operations. Stores context for autograd graph.
  """
  def __init__(self, *tensors: Tensor) -> None:
    self.parents = tensors
    self.saved_tensors: List[Any] = []

  def save_for_backward(self, *x: Any) -> None:
    """
    Save tensors needed for gradient computation.
    """
    self.saved_tensors.extend(x)

  def apply(self, *x: Any, **kwargs: Any) -> Tensor:
    """
    Apply the operation to input tensors.
    
    Sets up the computation graph for automatic differentiation.
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

def register(name: str, fxn: Any, gpu: bool = False) -> None:
  """
  Register an operation on the Tensor class to allow method chaining.
  
  Enables syntax like x.dot(w).relu() where w is a tensor.
  """
  if gpu:
    setattr(Tensor, name, lambda self, *x, **kwargs: fxn.apply(fxn, self, *x, **kwargs))
    Tensor.ops_gpu[name] = fxn 
  else:
    Tensor.ops[name] = fxn
  
  def dispatch(self: Tensor, *x: Any, **kwargs: Any) -> Tensor:
    op_func = (Tensor.ops_gpu if self.gpu else Tensor.ops)[name]
    return op_func.apply(op_func, self, *x, **kwargs)
  
  setattr(Tensor, name, dispatch)

  if name in ['add', 'sub', 'mul', 'div']:
    setattr(Tensor, "__%s__" % name, dispatch)
    setattr(Tensor, "__i%s__" % name, lambda self, x: self.assign(dispatch(self, x)))

import froog.ops # this registers all the operations
if GPU:
    # Import GPU operations based on available backend
    if DEVICE.name == "Metal":
        try:
            import froog.gpu.metal.ops_metal
        except ImportError:
            pass
    else:
        try:
            import froog.gpu.cl.ops_cl
        except ImportError:
            pass