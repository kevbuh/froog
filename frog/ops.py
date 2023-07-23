import numpy as np
from frog.tensor import Function, register
from frog.utils import im2col, col2im

# *********** Elementary Functions ***********
# Add, Mul, ReLU, Dot, Sum, Conv2D, Reshape 
# grad_output is the gradient of the loss with respect to the output of the operation.

class Add(Function):# x.add(y)
  @staticmethod     # @staticmethod doesn't require an instance of Add to work, so you can do x.add(y)
  def forward(ctx, x, y):
    return x + y
  
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output 
register("add", Add)

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


class ReLU(Function): # max(0,x)
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.maximum(input, 0)

  @staticmethod
  def backward(ctx, grad_output):
    (input,) = ctx.saved_tensors
    grad_input = grad_output * (input >= 0)
    return grad_input
register("relu", ReLU)


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

class Conv2D(Function): 
  @staticmethod
  def forward(ctx, input_image, conv_kernel):
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
    """
    ctx.save_for_backward(input_image, conv_kernel)
    cout, cin, H, W = conv_kernel.shape
    conv_to_return = np.zeros((input_image.shape[0], cout, input_image.shape[2]-(H-1), input_image.shape[3]-(W-1)), dtype=conv_kernel.dtype)

    tw = conv_kernel.reshape(conv_kernel.shape[0], -1).T                            # slice of kernel 

    for Y in range(conv_to_return.shape[2]):                                        # non_padded_height
      for X in range(conv_to_return.shape[3]):                                      # non_padded_width
        tx = input_image[:, :, Y:Y+H, X:X+W]
        tx = tx.reshape(input_image.shape[0], -1)
        conv_to_return[:, :, Y, X] = tx.dot(tw)
    return conv_to_return

  @staticmethod
  def backward(ctx, grad_output):
    x, conv_kernel = ctx.saved_tensors
    cout, cin, H, W = conv_kernel.shape
    dx, dw = np.zeros_like(x), np.zeros_like(conv_kernel)

    tw = conv_kernel.reshape(cout, -1)                                               # transformed kernel weights

    for Y in range(grad_output.shape[2]):
      for X in range(grad_output.shape[3]):
        gg = grad_output[:, :, Y, X]                                                 # backprop gradients from previous layer 
        tx = x[:, :, Y:Y+H, X:X+W].reshape(x.shape[0], -1)                           # slice of tensor at current conv op                                                                                # 
        dw += gg.T.dot(tx).reshape(dw.shape)                                         # gradient with respect to conv kernel
        dx[:, :, Y:Y+H, X:X+W] += gg.dot(tw).reshape(dx.shape[0], dx.shape[1], H, W) # accumulate gradient of input (current multiply element in chain rule)
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


class Reshape(Function):
  @staticmethod
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    return x.reshape(shape)

  @staticmethod
  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return grad_output.reshape(in_shape), None
register('reshape', Reshape)

class MaxPool2D(Function):
  @staticmethod
  def forward(ctx, x):
    my, mx = (x.shape[2]//2)*2, (x.shape[3]//2)*2         # ensures input tensor can be evenly divided into 2x2 blocks for max pooling
    stack = []
    cropped_x = x[:, :, :my, :mx]                         # crop input so 2x2 max pool can be taken
    for Y in range(2):
      for X in range(2):
        stack.append(cropped_x[:, :, Y::2, X::2][None])   # ::2 so 2x2 goes to next pool, [None] is numpy way to add an extra dimension so we can concatenate
    stack = np.concatenate(stack, axis=0)                 # put all into one row
    idx_of_max = np.argmax(stack, axis=0)
    ctx.save_for_backward(idx_of_max, x.shape)
    return np.max(stack, axis=0)

  @staticmethod
  def backward(ctx, grad_output):
    """
    Distributes the gradient from the output of the max pooling layer to its inputs
    The purpose of (idxs == (Y*2+X)) is to generate a boolean mask indicating the locations of the maximum values in each 2x2 block of the original input
    The expression (Y*2+X) is a way to iterate through the four possible positions within the 2x2 block: (0,0), (0,1), (1,0), and (1,1), which get mapped to the indices 0, 1, 2, and 3 
    """
    idxs, s = ctx.saved_tensors                                     
    my, mx = (s[2]//2)*2, (s[3]//2)*2                               # get shape that allows 2x2 max pool
    ret = np.zeros(s, dtype=grad_output.dtype)                      
    for Y in range(2):
      for X in range(2):
        ret[:, :, Y:my:2, X:mx:2] = grad_output * (idxs == (Y*2+X)) # selects the max and does the backward op
    return ret
register('max_pool2d', MaxPool2D)