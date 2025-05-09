# Overview

The most fundamental concept in all of ```froog``` and machine learning frameworks is the <a href="https://github.com/kevbuh/froog/blob/977b09caf32f21904768b08b2772139596604603/froog/tensor.py#L47">Tensor</a>. A <a href="https://en.wikipedia.org/wiki/Tensor_(machine_learning)">tensor</a> is simply a matrix of matrices (more accurately a multi-dimensional array). 

You can create a Tensor in ```froog``` with:
```python
import numpy as np
from froog.tensor import Tensor
my_tensor = Tensor([1,2,3])
```

Notice how we had to import NumPy. If you want to create a Tensor manually, make sure that it is a NumPy array!

<!-- Learn more about ```froog``` Tensors <a href="https://github.com/kevbuh/froog/blob/main/docs/tensors.md">here</a>. -->

# Creating a model

Okay cool, so now you know that ```froog```'s main datatype is a Tensor and uses NumPy in the background. How do I actually build a model? 

Here's an example of how to create an MNIST multi-layer perceptron (MLP). We wanted to make it as simple as possible for you to do so it resembles very basic Python concepts like classes. There are really only two methods you need to define: 
1. ```__init__``` that defines layers of the model (here we use ```Linear```) 
2. ```forward``` which defines how the input should flow through your model. We use a simple dot product with a ```Linear``` layer with a <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">```ReLU```</a> activation.

To create an instance of the ```mnistMLP``` model, do the same as you would in Python: ```model = mnistMLP()```. 

We support a few different optimizers, <a href="https://github.com/kevbuh/froog/blob/main/froog/optim.py">here</a> which include:
- <a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">Stochastic Gradient Descent (SGD)</a>
- <a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam">Adaptive Moment Estimation (Adam)</a>
- <a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp">Root Mean Square Propagation (RMSProp)</a>

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
    x.data = x.data.reshape((-1, 1, 28, 28))                     # get however many number of imgs in batch
    x = x.conv2d(self.c1).relu()                                 # pass through conv first
    x = x.reshape(shape=(x.shape[0], -1))
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()
```

So there are two quick examples to get you up and running. You might have noticed some operations like ```reshape``` and were wondering what else you can do with ```froog```. We have many more operations that you can apply on tensors: 