"""
Paper           : https://arxiv.org/abs/1905.11946
PyTorch version : https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

ConvNets are commonly developed at a fixed resource cost, and then scaled up in order to achieve better accuracy when more resources are made available

The scaling method was found by performing a grid search to find the relationship between different scaling dimensions of the baseline network under a fixed resource constraint

"SE" stands for "Squeeze-and-Excitation." Introduced by the "Squeeze-and-Excitation Networks" paper by Jie Hu, Li Shen, and Gang Sun (CVPR 2018).
"""
from froog import Tensor
from froog.ops import AvgPool2D, Conv2D
from froog.Tensor import Tensor

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
  Args:
    block_args    (namedtuple): BlockArgs, defined in utils.py.
    global_params (namedtuple): GlobalParam, defined in utils.py.
    image_size (tuple or list): [image_height, image_width].
  """
  def __init__(self, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio):
    
    # Expansion Phase (Inverted Bottleneck)
    oup = input_filters * expand_ratio

    self.se = se_ratio and 0 < se_ratio <= 1 # don't need?

    if expand_ratio != 1:
      self.expand_conv = Conv2D((oup, input_filters, 1,1))
      self._bn0 = BatchNorm2D((oup)) # TODO: how does work?

    # Depthwise convolution phase
    self._depthwise_conv = Conv2D((oup, oup, kernel_size, kernel_size)) # should be Tensor.zeros? Tensor.zeros(oup, 1, kernel_size, kernel_size)
    self._bn1 = BatchNorm2D(oup)

    # Squeeze and Excitation layer, if desired
    if self.has_se:
      num_squeezed_channels = max(1, int(input_filters * se_ratio))
      self._se_reduce = Tensor.zeros((num_squeezed_channels, oup, 1, 1))
      self._se_expand = Tensor.zeros((oup, num_squeezed_channels, 1, 1))

    # Pointwise convolution phase
    self._project_conv = Tensor.zeros((output_filters, oup, 1, 1))
    self._bn2 = BatchNorm2D(output_filters)
  

  def __call__(self, x):
    """
    Args:
      inputs (Tensor): Input Tensor.
      drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
    Returns:
      Output of this block after processing.
    """
    # Expansion and Depthwise Convolution
    if self._expand_conv:
      x = swish(self._bn0(x.conv2d(self._expand_conv)))
    x = x.pad2d(padding=(self.pad, self.pad, self.pad, self.pad)) # why needed?
    x = swish(self._depthwise_conv(x)._bn1(x))

    # Squeeze and Excitation
    if self.has_se:
      x_squeezed = AvgPool2D(x, kernel_size=(1,1))
      x_squeezed = self._se_reduce(x_squeezed)
      x_squeezed = swish(x_squeezed)
      x_squeezed = self._se_expand(x_squeezed)
      x = x_squeezed.sigmoid() * x

    # Pointwise Convolution
    x = self._project_conv(x)
    x = self._bn2(x)

    return x

class EfficientNet:
  def __init__(self):
    self._conv_stem = Tensor.zeros(32,3,3,3)  # in_channels, out_channels, k_h, k_w, need stride?
    self._bn0 = BatchNorm2D(32)
    self._fc = Tensor.zeros(1280,1000)
    self._fc_bias = Tensor.zeros(1000)        # TODO: what is this bias?

  def forward(self, x):
    x = AvgPool2D(x, kernel_size=(3,3))
    x = x.reshape(shape=(-1, 1280))
    x = x.dropout(0.2)
    return x.dot(self._fc).add(self._fc_bias) # TODO: why add bias?

if __name__ == "__main__":
  model = EfficientNet()
  # out = model.forward(Tensor.zeros(1, 3, 224, 224))
  # print(out)