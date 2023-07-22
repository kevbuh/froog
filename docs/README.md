# What is frog?

frog (Fast Real-time Optimization of Gradients) is an easy to read machine learning library. frog's driving philosophy is demanding simplicity in a world of complexity. Tensorflow and PyTorch are insanely complex with enormous codebases and meant for expert development.

Instead, frog is meant for those who are looking to get into machine learning, and want to actually understand how machine learning works before it is ultra optimized (which all modern ml libraries are).

### Where to start?

First, download frog using the <a href="https://github.com/kevbuh/frog/blob/main/docs/install.md">installation</a> docs. 

# Lets start building!

the most fundamental item in all of frog and machine learning is the Tensor. a tensor is simply a matrix of matrices (more accurately a multi-dimensional array). 

You can create a Tensor in frog by
```python
import numpy as np
from frog.tensor import Tensor

my_tensor = Tensor(np.array([1,2,3]))
```

notice how we had to import numpy. if you want to create a Tensor manually make sure that it is a Numpy array!

learn more about Tensors <a href="https://github.com/kevbuh/frog/blob/main/docs/tensors.md">here</a> 


### Actually creating something

Okay cool, so now you know that frog's main datatype is a Tensor and uses NumPy in the background. How do I actually build something? 

We wanted to make it as simple as possible for you to do so.

Heres an example of how to create an MNIST multi layer perceptron (MLP)

```python
from frog.tensor import Tensor
import frog.optim as optim

class mnistMLP:
  def __init__(self):
    self.l1 = Tensor(layer_init(784, 128))
    self.l2 = Tensor(layer_init(128, 10))

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
    self.c1 = Tensor(dense_layer(channels,1,conv_size,conv_size))     # (num_filters, color_channels, kernel_h, kernel_w)
    self.l1 = Tensor(dense_layer((28-conv_size+1)**2*channels, 128))  # (28-conv+1)(28-conv+1) since kernel isn't padded
    self.l2 = Tensor(dense_layer(128, 10))                            # MNIST output is 10 classes

  def forward(self, x):
    x.data = x.data.reshape((-1, 1, 28, 28))                          # get however many number of imgs in batch
    x = x.conv2d(self.c1).relu()                                      # pass through conv first
    x = x.reshape(Tensor(np.array((x.shape[0], -1))))
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()
```

### Tests

The tests are located <a href="https://github.com/kevbuh/frog/tree/main/tests">here</a>.

You can run them in your terminal by going into the root folder and entering

```
python tests/test_tensor.py
```

