# froog <img src="https://github.com/kevbuh/froog/actions/workflows/test.yml/badge.svg" alt="unit test badge" > <img src="https://static.pepy.tech/badge/froog" alt="num downloads badge">
<div align="center" >
  <img src="https://raw.githubusercontent.com/kevbuh/froog/main/assets/froog.png" alt="froog the frog" height="200">
  <br/>
  froog: fast real-time optimization of gradients 
  <br/>
  a beautifully compact machine-learning library
  <br/>
  <a href="https://github.com/kevbuh/froog">homepage</a> | <a href="https://github.com/kevbuh/froog/tree/main/docs">documentation</a> | <a href="https://pypi.org/project/froog/">pip</a>
  <br/>
  <br/>
</div>

<!--  froog is a SUPER SIMPLE machine learning framework with the goal of creating tools with AI, easily and efficiently. -->

```froog``` is an easy-to-read machine-learning library.

<!--  froog's driving philosophy is demanding simplicity in a world of complexity. -->

<!--  Tensorflow and PyTorch are insanely complex with enormous codebases and are meant for expert development. -->

```froog``` is meant for those looking to get into machine learning, who want to understand how the underlying machine learning framework's code works before they are ultra-optimized (which all modern ml libraries are).

```froog``` encapsulates everything from <a href="https://github.com/kevbuh/froog/blob/main/models/linear_regression.py">linear regression</a> to <a href="https://github.com/kevbuh/froog/blob/main/models/efficientnet.py">convolutional neural networks </a>

all of this in under 1000 lines. 

# Installation
```bash
pip install froog
```

More information on downloading ```froog``` in the <a href="https://github.com/kevbuh/froog/blob/main/docs/install.md">installation</a> docs. 


# Features
- <a href="https://github.com/kevbuh/froog/blob/main/froog/tensor.py">Custom Tensors</a> 
  - Backpropagation
  - Automatic Differentiation (autograd)
      - Forward and backward passes
- <a href="https://github.com/kevbuh/froog/blob/main/froog/ops.py">ML Operations</a> 
  - 2D Convolutions (im2col)
  - Numerical gradient checking
  - Acceleration methods (Adam)
  - Avg & Max pooling
- <a href="https://github.com/kevbuh/froog/blob/main/models/efficientnet.py">EfficientNet</a> inference
- <a href="https://github.com/kevbuh/froog/blob/main/froog/ops_gpu.py">GPU Support</a> 
- and a bunch <a href="https://github.com/kevbuh/froog/tree/main/froog">more</a> 

# Sneak Peek
```python
from froog.tensor import Tensor
from froog.nn import Linear
import froog.optim as optim

class mnistMLP:
  def __init__(self):
    self.l1 = Tensor(Linear(784, 128))
    self.l2 = Tensor(Linear(128, 10))

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = mnistMLP()
optim = optim.SGD([model.l1, model.l2], lr=0.001)
```

# Overview

The most fundamental concept in all of ```froog``` and machine learning is the Tensor. A <a href="https://en.wikipedia.org/wiki/Tensor_(machine_learning)">tensor</a> is simply a matrix of matrices (more accurately a multi-dimensional array). 

You can create a Tensor in ```froog``` by:
```python
import numpy as np
from froog.tensor import Tensor

my_tensor = Tensor([1,2,3])
```

Notice how we had to import numpy. If you want to create a Tensor manually make sure that it is a Numpy array!

Learn more about ```froog``` Tensors <a href="https://github.com/kevbuh/froog/blob/main/docs/tensors.md">here</a>.

### Actually creating something

Okay cool, so now you know that ```froog```'s main datatype is a Tensor and uses NumPy in the background. How do I actually build a model? 

We wanted to make it as simple as possible for you to do so.

Here's an example of how to create an MNIST multi-layer perceptron (MLP)

```python
from froog.tensor import Tensor
import froog.optim as optim
from froog.nn import Linear

class mnistMLP:
  def __init__(self):
    self.l1 = Tensor(Linear(784, 128))
    self.l2 = Tensor(Linear(128, 10))

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = mnistMLP()
optim = optim.SGD([model.l1, model.l2], lr=0.001)
```

You can also create a convolutional neural net by

```python
class SimpleConvNet:
  def __init__(self):
    conv_size = 5
    channels = 17
    self.c1 = Tensor(Linear(channels,1,conv_size,conv_size))     # (num_filters, color_channels, kernel_h, kernel_w)
    self.l1 = Tensor(Linear((28-conv_size+1)**2*channels, 128))  # (28-conv+1)(28-conv+1) since kernel isn't padded
    self.l2 = Tensor(Linear(128, 10))                            # MNIST output is 10 classes

  def forward(self, x):
    x.data = x.data.reshape((-1, 1, 28, 28))                          # get however many number of imgs in batch
    x = x.conv2d(self.c1).relu()                                      # pass through conv first
    x = x.reshape(shape=(x.shape[0], -1))
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()
```

# Contributing
<!-- THERES LOT OF STUFF TO WORK ON! VISIT THE <a href="https://github.com/kevbuh/froog/blob/main/docs/bounties.md">BOUNTY SHOP</a>  -->

Pull requests will be merged if they:
* increase simplicity
* increase functionality
* increase efficiency

More info on <a href="https://github.com/kevbuh/froog/blob/main/docs/contributing.md">contributing</a>

# Documentation

Need more information about how ```froog``` works? Visit the <a href="https://github.com/kevbuh/froog/tree/main/docs">documentation</a>.
