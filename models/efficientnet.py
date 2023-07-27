"""
Paper           : https://arxiv.org/abs/1905.11946
PyTorch version : https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

ConvNets are commonly developed at a fixed resource cost, and then scaled up in order to achieve better accuracy when more resources are made available

The scaling method was found by performing a grid search to find the relationship between different scaling dimensions of the baseline network under a fixed resource constraint

"SE" stands for "Squeeze-and-Excitation." Introduced by the "Squeeze-and-Excitation Networks" paper by Jie Hu, Li Shen, and Gang Sun (CVPR 2018).
"""
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
  
class MBConvBlock:    # Mobile Inverted Residual Bottleneck Block
  def __init__(self, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio):
    pass
    

  def __call__(self, x):
    pass

class EfficientNet:
  
  def __init__(self):
    pass

  def forward(self, x):
    pass
  

if __name__ == "__main__":
  model = EfficientNet()
  # out = model.forward(Tensor.zeros(1, 3, 224, 224))
  # print(out)