# inspired by pytorch
# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
# inspired by tinygrad

from functools import partialmethod
from inspect import signature
import numpy as np

# *********** Main Classes ***********
# ********* Tensor, Function *********
class Tensor:
  def __init__(self, data):
    if type(data) == list:
      data = np.array(data, dtype=np.float32)
    if type(data) != np.ndarray:
      print(f"error constructing tensor with {data}")
      assert False
    if data.dtype == np.float64:
      # print(f"sure you want to use float64 with {data}")
      pass
    self.data = data
    self.grad = None

    # internal variables used for autograd graph construction
    # these are where the backward gradient computation are saved
    self._ctx = None

  def __repr__(self):
      return f"Tensor data: {self.data}, gradients: {self.grad}" 

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

  def backward(self, allow_fill=True): # TODO: allow fill does what?
    if self._ctx is None:
      return

    if self.grad is None and allow_fill:
      # fill in the first grad with one
      assert self.data.size == 1
      self.grad = np.ones_like(self.data)

    assert self.grad is not None
      
    # autograd engine
    grads = self._ctx.backward(self._ctx, self.grad)
    if len(self._ctx.parents) == 1:
      grads = [grads]
    for t, g in zip(self._ctx.parents, grads):
      if g is None:
        continue
      if g.shape != t.data.shape:
        print(f"grad shape must match tensor shape in {self._ctx}, {g.shape} != {t.data.shape}")
        assert False
      t.grad = g
      t.backward(False) # what does inputting False do???

  def mean(self):
    div = Tensor(np.array([1 / self.data.size], dtype=self.data.dtype))
    return self.sum().mul(div)

class Function:
  """
  An instantiation of the Function class includes the context
  """
  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  def apply(self, arg, *x, **kwargs):
    """
    self  : is the tensor with data
    arg   : is the method  (.dot, .relu) 
    *x    : the input to the method  
    """
    if type(arg) == Tensor:
      op = self
      x = [arg]+list(x)
    else:
      op = arg
      x = [self]+list(x)
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

def register(name, fxn):
  """
  mechanism that allows you to chain methods in an intuitive and Pythonic way
  e.g. x.dot(w).relu(), where w is a tensor.

  partialmethod is used to create a new method that has some of the arguments to another method already filled in
  the apply method of that instance is added
  """
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))

import froog.ops # this registers all the operations