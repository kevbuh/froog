# What is FROG?

FROG (Fast Real-time Optimization of Gradients) is an easy to read machine learning library. FROG's driving philosophy is demanding simplicity in a world of complexity. Tensorflow and PyTorch are insanely complex with enormous codebases and meant for expert development.

Instead, FROG is meant for those who are looking to get into machine learning, and want to actually understand how machine learning works before it is ultra optimized (which all modern ml libraries are).

### Where to start?
The most fundamental item in all of FROG and machine learning is the Tensor. A tensor is simply a matrix of matrices (more accurately a multi-dimensional array). 

You can create a Tensor in FROG by
```python
import numpy as np
from frog.tensor import Tensor

my_tensor = Tensor(np.array([1,2,3]))
```

### Built with NumPy
Notice that we had to import NumPy. FROG is built with NumPy, which allows for general matrix operations. If you want to create a Tensor manually make sure that it is a Numpy array!

### Actually creating something

Okay cool, so now you know that FROG's main datatype is a Tensor and uses NumPy in the background. How do I actually build something? 

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