import numpy as np
from froog.tensor import Function, register
from froog.utils import im2col, col2im

# *********** Elementary Functions ***********
# Add, Mul, ReLU, Dot, Sum, Conv2D, Reshape 
# grad_output is the gradient of the loss with respect to the output of the operation.

# ******* core ops *******

class Add(Function):# x.add(y)
  @staticmethod     # @staticmethod doesn't require an instance of Add to work, so you can do x.add(y)
  def forward(ctx, x, y):
    return x + y
  
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output 
register("add", Add)

class Sub(Function): # x.sub(y)
  @staticmethod
  def forward(ctx, x, y):
    return x-y

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, -grad_output
register('sub', Sub)

class Mul(Function): # x.mul(y)
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x * y

  @staticmethod
  def backward(ctx, grad_output):
    x, y = ctx.saved_tensors
    return y * grad_output, x * grad_output
register("mul", Mul)

class Pow(Function): # x.pow(y)
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x ** y
  
  @staticmethod
  def backward(ctx, grad_output):
    x, y = ctx.saved_tensors
    return y * (x**(y-1.0)) * grad_output, (x**y) * np.log(x) * grad_output # power rule, d/dx (y^x)
register("pow", Pow)

class Dot(Function):  # x.dot(y)
  @staticmethod
  def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)
    return input.dot(weight)

  @staticmethod
  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = grad_output.dot(weight.T)
    grad_weight = grad_output.T.dot(input).T
    return grad_input, grad_weight
register('dot', Dot)

class Sum(Function):
  """
  reduce op
  reduces its input tensor to a single value by summing all the elements
  """
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.array([input.sum()])

  @staticmethod
  def backward(ctx, grad_output):
    (input,) = ctx.saved_tensors
    return grad_output * np.ones_like(input)
register("sum", Sum)

# ******* nn ops *******

class ReLU(Function): 
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.maximum(input, 0) # relu(x) = max(0,x)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output * (input >= 0)
    return grad_input
register("relu", ReLU)

class Sigmoid(Function): 
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    with np.warnings.catch_warnings():           # TODO: stable sigmoid? does the overflow matter?
      np.warnings.filterwarnings('ignore')
      ret = 1/(1 + np.exp(-input))               # sigmoid(x) = 1 / (1 + exp(-x))
    return ret 

  @staticmethod
  def backward(ctx, grad_output):
    ret, = ctx.saved_tensors
    grad_input = grad_output * (ret * (1 - ret)) # just take the derivative of sigmoid
    return grad_input
register("sigmoid", Sigmoid)

class Reshape(Function):
  @staticmethod
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    return x.reshape(shape)

  @staticmethod
  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return grad_output.reshape(in_shape)
register('reshape', Reshape)

class Pad2D(Function):
  """
  The first element (0,0) corresponds to padding along the batch dimension, which indicates no padding on both sides (0 elements added).
  """
  @staticmethod
  def forward(ctx, x, padding=None): 
    return np.pad(x, ((0,0), (0,0), (padding[0], padding[1]), (padding[2], padding[3]))) # (top, bottom, left, right)

  @staticmethod
  def backward(ctx, grad_output):
    raise Exception("write this")
register('pad2d', Pad2D)

class LogSoftmax(Function):
  """
  converts a vector of numbers into a vector of probabilities
  probabilities of each value are proportional to the scale of each value 
  """
  @staticmethod
  def forward(ctx, input):
    def logsumexp(x):
      c = x.max(axis=1)
      return c + np.log(np.exp(x - c.reshape((-1, 1))).sum(axis=1)) # axis=1 refers to the columns

    output = input - logsumexp(input).reshape((-1, 1))
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    (output,) = ctx.saved_tensors
    return grad_output - np.exp(output) * grad_output.sum(axis=1).reshape((-1, 1))
register("logsoftmax", LogSoftmax)

# ************* conv ops *************

class Conv2D(Function): 
  @staticmethod
  def forward(ctx, x, w, stride=1, groups=1):
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
    if type(ctx.stride) == int:                                                          # ctx stores function params
      ctx.stride = (ctx.stride, ctx.stride)

    cout, cin, H, W = w.shape

    if groups > 1:                                                                       # allows grouped convolutions 
      w = np.repeat(w, groups, axis=1) / groups                                          # TODO: why does this work?

    tw = w.reshape(cout, -1).T                                                           # slice of kernel 
    y_stride, x_stride = ctx.stride
    bs, oy, ox = x.shape[0], (x.shape[2]-(H-y_stride))//y_stride, (x.shape[3]-(W-x_stride))//x_stride
    ctx.save_for_backward(x, w)

    ret = np.zeros((bs, cout, oy, ox), dtype=w.dtype)

    for Y in range(oy):                                                                  # non_padded_height of output
      for X in range(ox):                                                                # non_padded_width  of output
        iY, iX = Y*y_stride, X*x_stride                                    
        tx = x[:, :, iY:iY+H, iX:iX+W].reshape(bs, -1)
        ret[:, :, Y, X] = tx.dot(tw)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    x, conv_kernel = ctx.saved_tensors
    cout, cin, H, W = conv_kernel.shape
    dx, dw = np.zeros_like(x), np.zeros_like(conv_kernel)
    tw = conv_kernel.reshape(cout, -1)                                                   # transformed kernel weights
    y_stride, x_stride = ctx.stride

    for Y in range(grad_output.shape[2]):
      for X in range(grad_output.shape[3]):
        iY,iX = Y*y_stride, X*x_stride
        gg = grad_output[:, :, Y, X]                                                     # current multiply element in chain rule
        tx = x[:, :, iY:iY+H, iX:iX+W].reshape(x.shape[0], -1)                           # slice of tensor at current conv op                                                                                # 
        dw += gg.T.dot(tx).reshape(dw.shape)                                             # gradient with respect to input 
        dx[:, :, iY:iY+H, iX:iX+W] += gg.dot(tw).reshape(dx.shape[0], dx.shape[1], H, W) # accumulate gradient with respect to weights 
    return dx, dw
register('conv2d', Conv2D)


class im2ColConv(Function):
  """
  uses im2col
  https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
  """

  @staticmethod
  def forward(ctx, x, w):
    cout, cin, k_h, k_x = w.shape
    bs, oy, ox = x.shape[0], x.shape[2]-(k_h-1), x.shape[3]-(k_x-1)
    tw = w.reshape(cout, -1).T                                             # each filter flattened into a row
    tx = im2col(x, k_h, k_x)                                               # im2col, turn input into column
    ctx.save_for_backward(tx, w)                                           # save the im2col output
    ret = tx.dot(tw).reshape(bs, oy, ox, cout)                             # now the conv has been transoforned into a GEMM
    return np.moveaxis(ret, [0,1,2,3], [0,2,3,1])                          # reorders the axes (batch size, number of channels, height, width)

  @staticmethod
  def backward(ctx, grad_output):
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

# ************* pooling ops *************

def stack_for_pool(x, pool_y, pool_x):
  my, mx = (x.shape[2]//pool_y)*pool_y, (x.shape[3]//pool_x)*pool_x        # ensures input tensor can be evenly divided into 2x2 blocks for max pooling
  stack = []
  cropped_x = x[:, :, :my, :mx]                                            # crop input so 2x2 max pool can be taken
  for Y in range(pool_y):
      for X in range(pool_x):
        stack.append(cropped_x[:, :, Y::pool_y, X::pool_x][None])          # ::2 so 2x2 goes to next pool, [None] is numpy way to add an extra dimension so we can concatenate
  return np.concatenate(stack, axis=0)                                     # put all into one row


def unstack_for_pool(fxn, s, py, px):
  max_y, max_x = (s[2]//py)*py, (s[3]//px)*px                               # get shape that allows (pool_size_y,pool_size_x) max pool
  for Y in range(py):
    for X in range(px):
      level_w_new_grad = fxn(Y*px+X)
      if X == 0 and Y == 0:                                                 # pool of zero size
        ret = np.zeros(s, dtype=level_w_new_grad.dtype)
      ret[:, :, Y:max_y:py, X:max_x:px] = level_w_new_grad
  return ret


class MaxPool2D(Function):
  @staticmethod
  def forward(ctx, x, kernel_size=(2,2)):
    stack = stack_for_pool(x, *kernel_size)
    idx_of_max = np.argmax(stack, axis=0)
    ctx.save_for_backward(idx_of_max, x.shape)
    return np.max(stack, axis=0)

  @staticmethod
  def backward(ctx, grad_output):
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


class AvgPool2D(Function):
  @staticmethod
  def forward(ctx, x, kernel_size=(2,2)):
    stack = stack_for_pool(x, *kernel_size)
    ctx.save_for_backward(x.shape)
    return np.mean(stack, axis=0)

  @staticmethod
  def backward(ctx, grad_output):
    s, = ctx.saved_tensors
    py, px = ctx.kernel_size                                               # TODO: where does kernel_size come from?
    my, mx = (s[2]//py)*py, (s[3]//px)*px
    ret = np.zeros(s, dtype=grad_output.dtype)
    for Y in range(py):
      for X in range(px):
        ret[:, :, Y:my:py, X:mx:px] = grad_output / py / px # divide by avg of pool, e.g. for 2x2 pool /= 4
    return ret
register('avg_pool2d', AvgPool2D)
