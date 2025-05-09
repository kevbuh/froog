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
  'VIZ=1 python3 models/efficientnet.py <https://optional_image_url>'

Output: 
  ******************************
  inference 9.63 s

  imagenet class: 254
  prediction    : pug, pug-dog
  probability   : 0.96068513
  ******************************
"""

import io
import os
import sys
import json
import numpy as np
from froog.tensor import Tensor
from froog.utils import fetch
from froog.ops import swish, BatchNorm2D, DropoutLayer
from froog import get_device

# Check if GPU is available
HAS_GPU = get_device() is not None and get_device().name != "CPU"

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

    self._dropout = DropoutLayer(0.2)
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
    x = self._dropout(x)
    return x.dot(self._fc).add(self._fc_bias.reshape(shape=[1,-1]))

  
  def load_weights_from_torch(self):
    import torch
    b0 = fetch("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth")
    with open(b0, 'rb') as f:
      b0 = torch.load(io.BytesIO(f.read()))

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
      if HAS_GPU:
        mv.gpu_()

def processImage(url):
  with open(fetch(url), 'rb') as f:
    img = Image.open(io.BytesIO(f.read()))
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
  with open('assets/imagenet_classes.txt', 'r') as f:
    lbls = json.load(f)

  # inference
  import time
  st = time.time()
  if HAS_GPU:
    out = model.forward(Tensor(img).to_gpu()).cpu()
  else:
    out = model.forward(Tensor(img))

  print(out.data)

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
