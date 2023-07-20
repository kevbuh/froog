# inspired by pytorch
# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
# inspired by tinygrad

from functools import partialmethod
import numpy as np

# *********** Main Classes ***********
# ********* Tensor, Function *********
class Tensor:
  def __init__(self, data):
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

  def backward(self, allow_fill=True):
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
      t.backward(False)

  def mean(self):
    # TODO: why taking mean? 
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

  def apply(self, arg, *x):
    """
    note that due to how partialmethod works, self and arg are switched
    self is the tensor                   (a)
    arg is the method                    (.dot, .relu) 
    *x is b --> the input to the method  (a.dot(b), a.add(b))
    support the args in both orders
    """
    if type(arg) == Tensor:
      op = self
      x = [arg]+list(x)
    else:
      op = arg
      x = [self]+list(x)
    ctx = op(*x)
    ret = Tensor(op.forward(ctx, *[t.data for t in x]))
    ret._ctx = ctx
    return ret

def register(name, fxn):
  """
  mechanism that allows you to chain methods in an intuitive and Pythonic way
  e.g. x.dot(w).relu(), where w is a tensor.
  """
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))

# this registers all the operations
import frog.ops