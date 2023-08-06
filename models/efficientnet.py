"""
 _______  _______  _______  ___  _______  ___  _______  __    _  _______       __    _  _______  _______ 
|       ||       ||       ||   ||       ||   ||       ||  |  | ||       |     |  |  | ||       ||       |
|    ___||    ___||    ___||   ||       ||   ||    ___||   |_| ||_     _|     |   |_| ||    ___||_     _|
|   |___ |   |___ |   |___ |   ||       ||   ||   |___ |       |  |   |       |       ||   |___   |   |  
|    ___||    ___||    ___||   ||      _||   ||    ___||  _    |  |   |       |  _    ||    ___|  |   |  
|   |___ |   |    |   |    |   ||     |_ |   ||   |___ | | |   |  |   |       | | |   ||   |___   |   |  
|_______||___|    |___|    |___||_______||___||_______||_|  |__|  |___|       |_|  |__||_______|  |___|  

Paper           : https://arxiv.org/abs/1905.11946
PyTorch version : https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

ConvNets are commonly developed at a fixed resource cost, and then scaled up in order to achieve better accuracy when more resources are made available
The scaling method was found by performing a grid search to find the relationship between different scaling dimensions of the baseline network under a fixed resource constraint
"SE" stands for "Squeeze-and-Excitation." Introduced by the "Squeeze-and-Excitation Networks" paper by Jie Hu, Li Shen, and Gang Sun (CVPR 2018).

Environment Variables:
  VIZ=1 --> plots processed image and output probabilities

How to Run:
  <VIZ=1> python models/efficientnet.py <https://image_url>

EfficientNet Hyper-Parameters and Weights:
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

import io
import os
import sys
import json
import numpy as np
from froog.tensor import Tensor
from froog.utils import fetch
from froog.nn import swish, BatchNorm2D

GPU = os.getenv("GPU", None) is not None

class MBConvBlock: 
  """
   Mobile Inverted Residual Bottleneck Block
  """
  def __init__(self, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio):
    
    # Expansion Phase (Inverted Bottleneck)
    oup = input_filters * expand_ratio
    if expand_ratio != 1:
      self._expand_conv = Tensor.zeros(oup, input_filters, 1, 1)
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
    x = x.conv2d(self._depthwise_conv, stride=self.strides, groups=self._depthwise_conv.shape[0])
    x = self._bn1(x)
    x = swish(x)

    # Squeeze and Excitation
    x_squeezed = x.avg_pool2d(kernel_size=x.shape[2:4])           # actual paper uses adaptive pool
    x_squeezed = swish(x_squeezed.conv2d(self._se_reduce).add(self._se_reduce_bias.reshape(shape=[1, -1, 1, 1])))
    x_squeezed = x_squeezed.conv2d(self._se_expand).add(self._se_expand_bias.reshape(shape=[1, -1, 1, 1]))
    x = x.mul(x_squeezed.sigmoid())

    # Pointwise Convolution
    x = x.conv2d(self._project_conv)
    x = self._bn2(x)

    if x.shape == inputs.shape:
      x = x.add(inputs)
    return x     

class EfficientNet:
  """
  blocks_args: [[num_repeats, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio]]
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
      for _ in range(block_arg[0]):                   # num times to repeat block
        self._blocks.append(MBConvBlock(*args))
        args[3] = args[4]                             # why do this
        args[1] = (1,1)
    
    # Head
    self._conv_head = Tensor.zeros(1280,320,1,1)      # TODO: why 320?
    self._bn1 = BatchNorm2D(1280)

    # self._dropout = Dropout(0.2)                    # TODO: make dropout layer
    self._fc = Tensor.zeros(1280, 1000)
    self._fc_bias = Tensor.zeros(1000)        

  def forward(self, x):
    x = x.pad2d(padding=(0,1,0,1))
    x = x.conv2d(self._conv_stem, stride=2)
    x = swish(self._bn0(x))

    for block in self._blocks:
      x = block(x)
    
    x = swish(self._bn1(x.conv2d(self._conv_head)))
    x = x.avg_pool2d(kernel_size=x.shape[2:4])
    x = x.reshape(shape=(-1, 1280))
    #x = x.dropout(0.2) # TODO: make dropout layers
    return x.dot(self._fc).add(self._fc_bias)

  
  def load_weights_from_torch(self): # TODO: what does eval do 
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
      if GPU:
        mv.gpu_()

def processImage(url):
  img = Image.open(io.BytesIO(fetch(url)))
  aspect_ratio = img.size[0] / img.size[1]
  img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))     
  img = np.array(img)
  y0,x0=(np.asarray(img.shape)[:2]-224)//2
  img = img[y0:y0+224, x0:x0+224]
  img = np.moveaxis(img, [2,0,1], [0,1,2])                    # (height, width, channels) --> (channels, height, width)
  if img.shape[0] == 4:                                       # RGB if image has transparency channel
    img = img[0:3,:,:]
  img = img.astype(np.float32).reshape(1,3,224,224)
  
  # normalize image for pretrained model
  img /= 255.0                                                # scales the pixel values from [0, 256) to [0, 1)
  img -= np.array([0.485, 0.456, 0.406]).reshape((1,-1,1,1))  # The values 0.485, 0.456, and 0.406 are the means of the red, green, and blue channels, respectively, of the ImageNet dataset.
  img /= np.array([0.229, 0.224, 0.225]).reshape((1,-1,1,1))  # The values 0.229, 0.224, and 0.225 are the standard deviations of the red, green, and blue channels, respectively, of the ImageNet dataset.
  return img

if __name__ == "__main__":
  # instantiate and get weights
  model = EfficientNet()
  model.load_weights_from_torch() 

  # load image and preprocess
  from PIL import Image
  if len(sys.argv) > 1:                                       
    url = sys.argv[1]
  else:
    url = "https://cdn.britannica.com/34/233234-050-1649BFA9/Pug-dog.jpg"

  # process image for pretrained weights
  img = processImage(url)

  if os.getenv('VIZ') == "1":
    import matplotlib.pyplot as plt
    plt.imshow(img[0].mean(axis=0))
    plt.show()

  # get imagenet labels into dictionary
  with open('datasets/imagenet_classes.txt', 'r') as f:
    lbls = json.load(f)

  # inference
  import time
  st = time.time()
  if GPU:
    out = model.forward(Tensor(img).to_gpu())
  else:
    out = model.forward(Tensor(img))

  if os.getenv('VIZ') == "1":
    # outputs
    import matplotlib.pyplot as plt
    plt.plot(out.data[0])
    plt.show()

  print("\n******************************")
  print(f"inference {float(time.time()-st):.2f} s\n")
  print("imagenet class:", np.argmax(out.data))
  print("prediction    :", lbls.get(str(np.argmax(out.data))))
  print("probability   :", np.max(out.data) / 10)
  print("******************************\n")