"""
Paper           : https://arxiv.org/abs/1905.11946
PyTorch version : https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

ConvNets are commonly developed at a fixed resource cost, and then scaled up in order to achieve better accuracy when more resources are made available
The scaling method was found by performing a grid search to find the relationship between different scaling dimensions of the baseline network under a fixed resource constraint
"SE" stands for "Squeeze-and-Excitation." Introduced by the "Squeeze-and-Excitation Networks" paper by Jie Hu, Li Shen, and Gang Sun (CVPR 2018).
"""
from froog import Tensor
from froog.ops import AvgPool2D
from froog.tensor import Tensor

def swish(x):
  return x.mul(x.sigmoid())

class BatchNorm2D:
  def __init__(self, sz):
    self.weight = Tensor.zeros(sz)
    self.bias = Tensor.zeros(sz)
    # TODO: need running_mean and running_var

  def __call__(self, x):
    return x

class MBConvBlock: # Mobile Inverted Residual Bottleneck Block
  """
   Mobile Inverted Residual Bottleneck Block
  """
  def __init__(self, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio):
    # Expansion Phase (Inverted Bottleneck)
    oup = input_filters * expand_ratio
    # self.has_se = se_ratio and 0 < se_ratio <= 1 # don't need?

    if expand_ratio != 1:
      self.expand_conv = Tensor.zeros(oup, input_filters, 1, 1)
      self._bn0 = BatchNorm2D(oup)          

    # Depthwise convolution phase
    self._depthwise_conv = Tensor.zeros(oup, oup, kernel_size, kernel_size)
    self._bn1 = BatchNorm2D(oup)

    # Squeeze and Excitation (SE) layer
    num_squeezed_channels = max(1, int(input_filters * se_ratio))
    self._se_reduce = Tensor.zeros(num_squeezed_channels, oup, 1, 1)
    self._se_expand = Tensor.zeros(oup, num_squeezed_channels, 1, 1)
    self._se_reduce_bias = Tensor.zeros(num_squeezed_channels)

    # Pointwise convolution phase
    self._project_conv = Tensor.zeros(output_filters, oup, 1, 1)
    self._bn2 = BatchNorm2D(output_filters)
  
  def __call__(self, inputs): # TODO: what is __call__ 
    """
    Args:
      inputs (Tensor): Input Tensor.
      drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
    Returns:
      Output of this block after processing.
    """
    # Expansion and Depthwise Convolution
    x = inputs
    if self._expand_conv:
      x = swish(self._bn0(x.conv2d(self._expand_conv)))
    x = x.pad2d(padding=(self.pad, self.pad, self.pad, self.pad)) # why needed?
    x = self._depthwise_conv(x)
    x = self._bn1(x)
    x = swish(x)

    # Squeeze and Excitation
    if self.has_se:
      x_squeezed = AvgPool2D(x, kernel_size=x.shape[2:4])  # TODO: what is adaptive avg pool? 
      x_squeezed = self._se_reduce(x_squeezed)
      x_squeezed = swish(x_squeezed)
      x_squeezed = x_squeezed.add(self._se_reduce_bias).reshape(shape=[1, -1, 1, 1])
      x_squeezed = self._se_expand(x_squeezed)
      x = x.mul(x_squeezed.sigmoid())

    # Pointwise Convolution
    x = x.conv2d(self._project_conv)
    x = x.conv2d(self._bn2)

    if x.shape == inputs.shape:
      x = x.add(inputs)

    # TODO: what is drop connect?
    return x

class EfficientNet:
  def __init__(self):
    self._conv_stem = Tensor.zeros(32,3,3,3)  # in_channels, out_channels, k_h, k_w, need stride?
    self._bn0 = BatchNorm2D(32)
    block_args = []
    self._blocks = []
    # TODO: build blocks
    for block_arg in block_args:
      pass
    
    # Head
    self._conv_head = Tensor.zeros(1280,320,1,1) # why 320?
    self._bn1 = BatchNorm2D(1280)

    # Final linear layer
    self._avg_pooling = AvgPool2D(1)
    # self._dropout = Dropout(0.2) # TODO: make dropout layer
    self._fc = Tensor.zeros(1280, 1000)
    self._fc_bias = Tensor.zeros(1000)        # TODO: what is this bias?

  def forward(self, x):
    x = AvgPool2D(x, kernel_size=(1,1))
    x = x.reshape(shape=(-1, 1280))
    # x = x.dropout(0.2)
    return x.dot(self._fc).add(self._fc_bias) # TODO: why add bias?

if __name__ == "__main__":
  model = EfficientNet()
  # out = model.forward(Tensor.zeros(1, 3, 224, 224))
  # print(out)