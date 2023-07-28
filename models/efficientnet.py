"""
Paper           : https://arxiv.org/abs/1905.11946
PyTorch version : https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

ConvNets are commonly developed at a fixed resource cost, and then scaled up in order to achieve better accuracy when more resources are made available
The scaling method was found by performing a grid search to find the relationship between different scaling dimensions of the baseline network under a fixed resource constraint
"SE" stands for "Squeeze-and-Excitation." Introduced by the "Squeeze-and-Excitation Networks" paper by Jie Hu, Li Shen, and Gang Sun (CVPR 2018).

go to bottom of file to see params and weights
"""
from froog.tensor import Tensor
from froog.ops import AvgPool2D
from froog.utils import fetch
from froog.nn import swish, BatchNorm2D
import io
import sys
import numpy as np


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
    else: 
      self._expand_conv = None

    self.pad = (kernel_size-1)//2
    self.strides = strides      

    # Depthwise convolution phase
    self._depthwise_conv = Tensor.zeros(oup, 1, kernel_size, kernel_size)
    self._bn1 = BatchNorm2D(oup)

    # Squeeze and Excitation (SE) layer
    num_squeezed_channels = max(1, int(input_filters * se_ratio))
    self._se_reduce = Tensor.zeros(num_squeezed_channels, oup, 1, 1)
    self._se_reduce_bias = Tensor.zeros(num_squeezed_channels)
    self._se_expand = Tensor.zeros(oup, num_squeezed_channels, 1, 1)
    self._se_expand_bias = Tensor.zeros(oup)

    # Pointwise convolution phase
    self._project_conv = Tensor.zeros(output_filters, oup, 1, 1)
    self._bn2 = BatchNorm2D(output_filters)
  
  def __call__(self, inputs):                                     # __call__ allows instance of class to be called as a function
    # Expansion and Depthwise Convolution
    x = inputs
    if self._expand_conv:
      x = swish(self._bn0(x.conv2d(self._expand_conv)))
    x = x.pad2d(padding=(self.pad, self.pad, self.pad, self.pad)) # maintain same size ouput as input after conv operation
    x = self._depthwise_conv(x)
    x = self._bn1(x)
    x = swish(x)

    # Squeeze and Excitation
    if self.has_se:
      x_squeezed = AvgPool2D(x, kernel_size=x.shape[2:4])         # actual paper uses adaptive avg pool
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

    return x     

class EfficientNet:
  """
  blocks_args: (num_repeats, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio)
  """
  def __init__(self):
    self._conv_stem = Tensor.zeros(32,3,3,3)          # in_channels, out_channels, k_h, k_w, need stride?
    self._bn0 = BatchNorm2D(32)
    block_args = [
      [1, 3, (1,1), 1, 32, 16, 0.25],
      [2, 3, (2,2), 6, 16, 24, 0.25],
      [2, 5, (2,2), 6, 24, 40, 0.25],
      [3, 3, (2,2), 6, 40, 80, 0.25],
      [3, 5, (1,1), 6, 80, 112, 0.25],
      [4, 5, (2,2), 6, 112, 192, 0.25],
      [1, 3, (1,1), 6, 192, 320, 0.25],
    ] 
    self._blocks = []
    
    for block_arg in block_args:                      
      args = block_arg[1:]
      for n in range(block_arg[0]):                   # num times to repeat block
        self._blocks.append(MBConvBlock(*args))
        args[3] = args[4]                             # why do this
        args[1] = (1,1)
    
    # Head
    self._conv_head = Tensor.zeros(1280,320,1,1)      # TODO: why 320?
    self._bn1 = BatchNorm2D(1280)

    # Final linear layer
    self._avg_pooling = AvgPool2D(1)
    # self._dropout = Dropout(0.2)                    # TODO: make dropout layer
    self._fc = Tensor.zeros(1280, 1000)
    self._fc_bias = Tensor.zeros(1000)        

  def forward(self, x):
    x = AvgPool2D(x, kernel_size=(1,1))
    x = x.reshape(shape=(-1, 1280))
    # x = x.dropout(0.2)
    return x.dot(self._fc).add(self._fc_bias) 
  
  def load_weights_from_torch(self): # ???? 
    # load b0
    import torch
    b0 = fetch("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth")
    b0 = torch.load(io.BytesIO(b0))

    for k,v in b0.items():
      if '_blocks.' in k:
        k = "%s[%s].%s" % tuple(k.split(".", 2))
      mk = "self."+k
      try:
        mv = eval(mk)
      except AttributeError:
        try:
          mv = eval(mk.replace(".weight", ""))
        except AttributeError:
          mv = eval(mk.replace(".bias", "_bias"))
      vnp = v.numpy().astype(np.float32)
      mv.data[:] = vnp if k != '_fc.weight' else vnp.T        # assigns data to enet

if __name__ == "__main__":
  # instantiate and get weights
  model = EfficientNet()
  model.load_weights_from_torch() 

  # load image and preprocess
  from PIL import Image
  if len(sys.argv) > 1:                                       # for different url
    url = sys.argv[1]
  else:
    url = "https://c.files.bbci.co.uk/12A9B/production/_111434467_gettyimages-1143489763.jpg"

  img = Image.open(io.BytesIO(fetch(url)))
  aspect_ratio = img.size[0] / img.size[1]
  img = img.resize((int(224*aspect_ratio), 224))              # resizes height to 224 pixels and retains aspect ratio
  img = np.array(img)

  crop = (img.shape[1]-224)//2                                # crop img
  img = img[:, crop:crop+224]
  img = np.moveaxis(img, [2,0,1], [0,1,2])                    # (height, width, channels) --> (channels, height, width)
  img = img.astype(np.float32).reshape(1,3,224,224)

  # normalize for pretrained
  img /= 256                                                  # scales the pixel values from [0, 256) to [0, 1)
  img -= np.array([0.485, 0.456, 0.406]).reshape((1,-1,1,1))  # The values 0.485, 0.456, and 0.406 are the means of the red, green, and blue channels, respectively, of the ImageNet dataset.
  img /= np.array([0.229, 0.224, 0.225]).reshape((1,-1,1,1))  # The values 0.229, 0.224, and 0.225 are the standard deviations of the red, green, and blue channels, respectively, of the ImageNet dataset.

  # import matplotlib.pyplot as plt
  # plt.imshow(img[0].mean(axis=0))
  # plt.show()

  # get imagenet labels into dictionary
  import ast
  lbls = fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
  lbls = ast.literal_eval(lbls.decode('utf-8'))

  # run the net
  import time
  st = time.time()
  out = model.forward(Tensor(img))

  # if you want to look at the outputs
  """
  import matplotlib.pyplot as plt
  plt.plot(out.data[0])
  plt.show()
  """

  print(f"did inference in {float(time.time()-st):.2f} s" )
  print(np.argmax(out.data), np.max(out.data), lbls[np.argmax(out.data)])


"""
EfficientNet b0-7 Params and Weights
url_map = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}

params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }

blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
"""