#  _______  ______    _______  _______  _______ 
# |       ||    _ |  |       ||       ||       |
# |    ___||   | ||  |   _   ||   _   ||    ___|
# |   |___ |   |_||_ |  | |  ||  | |  ||   | __ 
# |    ___||    __  ||  |_|  ||  |_|  ||   ||  |
# |   |    |   |  | ||       ||       ||   |_| |
# |___|    |___|  |_||_______||_______||_______|

import numpy as np
from typing import Tuple, List, Union, Optional, Any, TypeVar, cast, Callable
from froog.tensor import UOP, register
from froog.utils import im2col, col2im
from froog.tensor import Tensor

# *****************************************************
#     ____  ___   _____ __________   ____  ____  _____
#    / __ )/   | / ___//  _/ ____/  / __ \/ __ \/ ___/
#   / __  / /| | \__ \ / // /      / / / / /_/ /\__ \ 
#  / /_/ / ___ |___/ // // /___   / /_/ / ____/___/ / 
# /_____/_/  |_/____/___/\____/   \____/_/    /____/  
#
# **************** Basic Operations ***************
# - grad_output is the gradient of the loss with respect to the output of the operation.

class Add(UOP):# x.add(y)
  @staticmethod     # @staticmethod doesn't require an instance of Add to work, so you can do x.add(y)
  def forward(ctx: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y
  
  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return grad_output, grad_output 
register("add", Add)

class Sub(UOP): # x.sub(y)
  @staticmethod
  def forward(ctx: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x-y

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return grad_output, -grad_output
register('sub', Sub)

class Mul(UOP): # x.mul(y)
  @staticmethod
  def forward(ctx: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(x, y)
    return x * y

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, y = ctx.saved_tensors
    return y * grad_output, x * grad_output
register("mul", Mul)

class Sum(UOP): # x.sum()
  """
  reduce op
  reduces its input tensor to a single value by summing all the elements
  """
  @staticmethod
  def forward(ctx: Any, input: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(input)
    return np.array([input.sum()])

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> np.ndarray:
    (input,) = ctx.saved_tensors
    return grad_output * np.ones_like(input)
register("sum", Sum)

class Pow(UOP): # x.pow(y)
  @staticmethod
  def forward(ctx: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(x, y)
    return x ** y
  
  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, y = ctx.saved_tensors
    return y * (x**(y-1.0)) * grad_output, (x**y) * np.log(x) * grad_output # power rule, d/dx (y^x)
register("pow", Pow)


# *******************************               
#    ______________  _____  ___
#   / ____/ ____/  |/  /  |/  /
#  / / __/ __/ / /|_/ / /|_/ / 
# / /_/ / /___/ /  / / /  / /  
# \____/_____/_/  /_/_/  /_/   
# 
# ******* GEMM ops *******          
                             
class Dot(UOP):  # x.dot(y)
  @staticmethod
  def forward(ctx: Any, input: np.ndarray, weight: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(input, weight)
    return input.dot(weight)

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    input, weight = ctx.saved_tensors
    grad_input = grad_output.dot(weight.T)
    grad_weight = input.T.dot(grad_output)
    return grad_input, grad_weight
register('dot', Dot)
register('matmul', Dot)


# ***********************************************************
#    _____ ______  _______  __    ______   ____  ____  _____
#   / ___//  _/  |/  / __ \/ /   / ____/  / __ \/ __ \/ ___/
#   \__ \ / // /|_/ / /_/ / /   / __/    / / / / /_/ /\__ \ 
#  ___/ // // /  / / ____/ /___/ /___   / /_/ / ____/___/ / 
# /____/___/_/  /_/_/   /_____/_____/   \____/_/    /____/  
#
# ************************ nn ops ***********************              
                                                          
class ReLU(UOP): 
  @staticmethod
  def forward(ctx: Any, input: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(input)
    return np.maximum(input, 0)                     # relu(x) = max(0,x)

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> np.ndarray:
    input, = ctx.saved_tensors
    grad_input = grad_output * (input >= 0)
    return grad_input
register("relu", ReLU)

class Sigmoid(UOP): 
  @staticmethod
  def forward(ctx: Any, input: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(input)
    ret = 1/(1 + np.exp(-input))                    # sigmoid(x) = 1 / (1 + exp(-x))
    return ret 

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> np.ndarray:
    ret, = ctx.saved_tensors
    grad_input = grad_output * (ret * (1 - ret))    # just take the derivative of sigmoid
    return grad_input
register("sigmoid", Sigmoid)

class DropoutLayer:
  """
  Dropout layer that randomly sets a fraction of input units to 0 during training time.
  pytorch version: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
  """
  def __init__(self, p: float = 0.5) -> None:
    self.p = p
    self.training = True

  def __call__(self, x: Tensor) -> Tensor:
    if not self.training or self.p == 0: return x
    from froog.tensor import Tensor
    mask = (np.random.rand(*x.data.shape) >= self.p).astype(np.float32) / (1.0 - self.p)
    return Tensor(x.data * mask, gpu=x.gpu)

class Dropout(UOP):
  @staticmethod
  def forward(ctx: Any, input: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    if not training: return input
    # create a binary mask with probability (1-p) of being 1
    # scale by 1/(1-p) to keep expectation same
    ctx.training = training
    mask = (np.random.rand(*input.shape) >= p).astype(np.float32) / (1.0 - p if p < 1.0 else 1e-9) # avoid division by zero if p is 1.0
    ctx.save_for_backward(mask)
    return input * mask

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> np.ndarray:
    if not ctx.training: return grad_output
    mask, = ctx.saved_tensors
    return grad_output * mask
register("dropout", Dropout)

class Reshape(UOP):
  @staticmethod
  def forward(ctx: Any, x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    ctx.save_for_backward(x.shape)
    return x.reshape(shape)

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> np.ndarray:
    in_shape, = ctx.saved_tensors
    return grad_output.reshape(in_shape)
register('reshape', Reshape)

class Pad2D(UOP):
  """
  The first element (0,0) corresponds to padding along the batch dimension, which indicates no padding on both sides (0 elements added).
  """
  @staticmethod
  def forward(ctx: Any, x: np.ndarray, padding: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray: 
    if padding is None:
      padding = (0, 0, 0, 0)
    return np.pad(x, ((0,0), (0,0), (padding[0], padding[1]), (padding[2], padding[3]))) # (top, bottom, left, right)

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> np.ndarray:
    raise Exception("write this")
register('pad2d', Pad2D)

class LogSoftmax(UOP):
  """
  converts a vector of numbers into a vector of probabilities
  probabilities of each value are proportional to the scale of each value 
  """
  @staticmethod
  def forward(ctx: Any, input: np.ndarray) -> np.ndarray:
    def logsumexp(x: np.ndarray) -> np.ndarray:
      c = x.max(axis=1)
      return c + np.log(np.exp(x - c.reshape((-1, 1))).sum(axis=1)) # axis=1 refers to the columns

    output = input - logsumexp(input).reshape((-1, 1))
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> np.ndarray:
    (output,) = ctx.saved_tensors
    return grad_output - np.exp(output)*(grad_output.sum(axis=1).reshape((-1, 1)))
register("logsoftmax", LogSoftmax)


# *************************************************
#    __________  _   ___    __   ____  ____  _____
#   / ____/ __ \/ | / / |  / /  / __ \/ __ \/ ___/
#  / /   / / / /  |/ /| | / /  / / / / /_/ /\__ \ 
# / /___/ /_/ / /|  / | |/ /  / /_/ / ____/___/ / 
# \____/\____/_/ |_/  |___/   \____/_/    /____/  
#
# ****************** conv ops *****************

class Conv2D(UOP): # TODO: understand group splits
  @staticmethod
  def forward(ctx: Any, x: np.ndarray, w: np.ndarray, stride: Union[int, Tuple[int, int]] = 1, groups: int = 1) -> np.ndarray:
    """
    https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    WARNING: doesn't handle padding or strides yet
    Args:
      x.shape[0] 									  --> number of input examples (batch size)
      cout 			 								    --> number of output channels
      x.shape[2]-(H-1)					 	  --> non-padded height of conv output, need to subtract because this is an unpadded conv
      x.shape[3]-(W-1)						  --> width of output
    Shape: 
      (a, b, c, d)(e, f, g, h)      --> (a, e, c-(g-1), d-(h-1)) 
      in general, output x and y = [(W−K+2P)/S]+1
    """
    ctx.stride = stride
    ctx.groups = groups
    
    if isinstance(ctx.stride, int):                                                                           # ctx stores function params
      ctx.stride = (ctx.stride, ctx.stride)

    cout, cin, H, W = w.shape

    tw = w.reshape(cout, -1).T                                                                            # slice of kernel 
    y_stride, x_stride = ctx.stride

    bs,cin_,oy,ox = x.shape[0], x.shape[1], (x.shape[2]-(H-y_stride))//y_stride, (x.shape[3]-(W-x_stride))//x_stride
    assert cin*ctx.groups == cin_                                                                         # ensures that the channel dimensions match appropriately for grouping
    assert cout % ctx.groups == 0                                                                         # ensures that the number of output channels can be evenly divided among the groups
    g_w_chans = cout//ctx.groups                                                                          # number of output channels per group

    ctx.save_for_backward(x, w)
    ret = np.zeros((bs, cout, oy, ox), dtype=w.dtype)
    
    for g in range(ctx.groups):
      tw = w[g*g_w_chans:(g*g_w_chans+g_w_chans)].reshape(g_w_chans, -1).T                                # transformed kernel weights
      for Y in range(oy):
        for X in range(ox):
          iY,iX = Y*y_stride, X*x_stride
          tx = x[:, g*cin:(g*cin+cin), iY:iY+H, iX:iX+W].reshape(bs, -1)
          ret[:, g*g_w_chans:(g*g_w_chans+g_w_chans), Y, X] += tx.dot(tw)
    return ret

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, w = ctx.saved_tensors
    cout, cin, H, W = w.shape
    dx, dw = np.zeros_like(x), np.zeros_like(w)                                         
    y_stride, x_stride = ctx.stride
    g_w_chans = cout//ctx.groups

    for g in range(ctx.groups):
      tw = w[g*g_w_chans:(g*g_w_chans+g_w_chans)].reshape(g_w_chans, -1)
      for Y in range(grad_output.shape[2]):
        for X in range(grad_output.shape[3]):
          iY,iX = Y*y_stride, X*x_stride
          gg = grad_output[:, g*g_w_chans:(g*g_w_chans+g_w_chans), Y, X]                                  # current multiply element in chain rule
          tx = x[:, g*cin:(g*cin+cin), iY:iY+H, iX:iX+W].reshape(x.shape[0], -1)                          # slice of tensor at current conv op        
          dw[g*g_w_chans:(g*g_w_chans+g_w_chans)] += gg.T.dot(tx).reshape((g_w_chans,cin,H,W))            # gradient with respect to input
          dx[:, g*cin:(g*cin+cin), iY:iY+H, iX:iX+W] += gg.dot(tw).reshape(dx.shape[0], cin, H, W)        # accumulate gradient with respect to weights 
    return dx, dw
register('conv2d', Conv2D)


class im2ColConv(UOP):
  """
  uses im2col
  https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
  """

  @staticmethod
  def forward(ctx: Any, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    cout, cin, k_h, k_x = w.shape
    bs, oy, ox = x.shape[0], x.shape[2]-(k_h-1), x.shape[3]-(k_x-1)
    tw = w.reshape(cout, -1).T                                             # each filter flattened into a row
    tx = im2col(x, k_h, k_x)                                               # im2col, turn input into column
    ctx.save_for_backward(tx, w)                                           # save the im2col output
    ret = tx.dot(tw).reshape(bs, oy, ox, cout)                             # now the conv has been transoforned into a GEMM
    return np.moveaxis(ret, [0,1,2,3], [0,2,3,1])                          # reorders the axes (batch size, number of channels, height, width)

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bs,_,oy,ox = grad_output.shape
    tx, w = ctx.saved_tensors                                              # transformed input, filter weights 
    cout,cin,H,W = w.shape
    tw = w.reshape(cout, -1)                                               # flatten filter and stack onto other channel filters
    gg = np.moveaxis(grad_output, [0,1,2,3], [1,0,2,3]).reshape(cout, -1)  # order correctly
    dw = gg.dot(tx).reshape(w.shape)                                       # compute gradient of weight
    dxi = gg.T.dot(tw)                                                     # compute gradient of input
    dx = col2im(dxi, H, W, oy+(H-1), ox+(W-1))                             # turn columns back into image shape
    return dx, dw
register('im2col2dconv', im2ColConv)


# *************************************************
#     ____  ____  ____  __       ____  ____  _____
#    / __ \/ __ \/ __ \/ /      / __ \/ __ \/ ___/
#   / /_/ / / / / / / / /      / / / / /_/ /\__ \ 
#  / ____/ /_/ / /_/ / /___   / /_/ / ____/___/ / 
# /_/    \____/\____/_____/   \____/_/    /____/  
#
# **************** pooling ops ***************

def stack_for_pool(x: np.ndarray, pool_y: int, pool_x: int) -> np.ndarray:
  my, mx = (x.shape[2]//pool_y)*pool_y, (x.shape[3]//pool_x)*pool_x        # ensures input tensor can be evenly divided into 2x2 blocks for max pooling
  stack = []
  cropped_x = x[:, :, :my, :mx]                                            # crop input so 2x2 max pool can be taken
  for Y in range(pool_y):
      for X in range(pool_x):
        stack.append(cropped_x[:, :, Y::pool_y, X::pool_x][None])          # ::2 so 2x2 goes to next pool, [None] is numpy way to add an extra dimension so we can concatenate
  return np.concatenate(stack, axis=0)                                     # put all into one row


def unstack_for_pool(fxn: Callable[[int], np.ndarray], s: Tuple[int, ...], py: int, px: int) -> np.ndarray:
  max_y, max_x = (s[2]//py)*py, (s[3]//px)*px                              # get shape that allows (pool_size_y,pool_size_x) max pool
  ret = None
  for Y in range(py):
    for X in range(px):
      level_w_new_grad = fxn(Y*px+X)
      if X == 0 and Y == 0:                                                # pool of zero size
        ret = np.zeros(s, dtype=level_w_new_grad.dtype)
      if ret is not None:
        ret[:, :, Y:max_y:py, X:max_x:px] = level_w_new_grad
  return ret if ret is not None else np.zeros(s)


class MaxPool2D(UOP):
  @staticmethod
  def forward(ctx: Any, x: np.ndarray, kernel_size: Tuple[int, int] = (2,2)) -> np.ndarray:
    ctx.kernel_size = kernel_size
    stack = stack_for_pool(x, *kernel_size)
    idx_of_max = np.argmax(stack, axis=0)
    ctx.save_for_backward(idx_of_max, x.shape)
    return np.max(stack, axis=0)

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> np.ndarray:
    """
    Distributes the gradient from the output of the max pooling layer to its inputs
    The purpose of (idxs == idx) is to generate a boolean mask indicating the locations of the maximum values in each 2x2 block of the original input
    The expression (Y*2+X) is a way to iterate through the four possible positions within the kernel block: e.g. (0,0), (0,1), (1,0), and (1,1), which get mapped to the indices 0, 1, 2, and 3 
    """
    idxs, s = ctx.saved_tensors
    return unstack_for_pool(lambda idx: grad_output * (idxs == idx), 
                            s,
                            *ctx.kernel_size)
register('max_pool2d', MaxPool2D)

class AvgPool2D(UOP):
  @staticmethod
  def forward(ctx: Any, x: np.ndarray, kernel_size: Tuple[int, int] = (2,2)) -> np.ndarray:
    ctx.kernel_size = kernel_size
    stack = stack_for_pool(x, *kernel_size)
    ctx.save_for_backward(x.shape)
    return np.mean(stack, axis=0)

  @staticmethod
  def backward(ctx: Any, grad_output: np.ndarray) -> np.ndarray:
    s, = ctx.saved_tensors
    py, px = ctx.kernel_size                                  # kernel_size passed from forward context
    my, mx = (s[2]//py)*py, (s[3]//px)*px
    ret = np.zeros(s, dtype=grad_output.dtype)
    for Y in range(py):
      for X in range(px):
        ret[:, :, Y:my:py, X:mx:px] = grad_output / py / px   # divide by avg of pool, e.g. for 2x2 pool /= 4
    return ret
register('avg_pool2d', AvgPool2D)

# *************************************
#     _   ___   __   ____  ____  _____
#    / | / / | / /  / __ \/ __ \/ ___/
#   /  |/ /  |/ /  / / / / /_/ /\__ \ 
#  / /|  / /|  /  / /_/ / ____/___/ / 
# /_/ |_/_/ |_/   \____/_/    /____/  
#
# ************* nn ops ************   

def Linear(*x: int) -> np.ndarray:
  # random Glorot initialization
  ret = np.random.uniform(-1., 1., size=x)/np.sqrt(np.prod(x))
  return ret.astype(np.float32)

def swish(x: Tensor) -> Tensor:
  return x.mul(x.sigmoid())

class BatchNorm2D:
  """
  __call__ follows the formula from the link below
  pytorch version: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

  self.weight       = γ
  self.bias         = β
  self.running_mean = E[x] 
  self.running_var  = Var[x]

  the reshaping step ensures that each channel of the input has its 
  own separate set of parameters (mean, variance, weight, and bias)

  self.running_mean has shape [num_channels].
  self.running_mean.reshape(shape=[1, -1, 1, 1]) reshapes it to [1, num_channels, 1, 1]
  """
  def __init__(self, sz: int, eps: float = 0.001) -> None:
    self.eps = eps
    self.weight = Tensor.zeros(sz)
    self.bias = Tensor.zeros(sz)

    # TODO: need running_mean and running_var
    self.running_mean = Tensor.zeros(sz)
    self.running_var = Tensor.zeros(sz)
    self.num_batches_tracked = Tensor.zeros(1)

  def __call__(self, x: Tensor) -> Tensor:
    x = x.sub(self.running_mean.reshape(shape=[1, -1, 1, 1]))
    x = x.mul(self.weight.reshape(shape=[1, -1, 1, 1]))
    x = x.div(self.running_var.add(Tensor([self.eps], gpu=x.gpu)).reshape(shape=[1, -1, 1, 1]).sqrt())
    x = x.add(self.bias.reshape(shape=[1, -1, 1, 1]))
    return x