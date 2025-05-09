# What is froog?

froog is an easy to read machine learning library. froog's driving philosophy is demanding simplicity in a world of complexity. 

Tensorflow and PyTorch are insanely complex with enormous codebases and meant for expert development. Instead, froog is meant for those who are looking to get into machine learning, and want to actually understand how machine learning works before it is ultra-optimized (which all modern ml libraries are).

# Installation 

simply install froog onto your local computer by entering the following into your terminal 

```bash
pip install froog
# OR
git clone https://github.com/kevbuh/froog.git
cd froog
pip3 install -r requirements.txt
```

# Overview

The most fundamental concept in all of `froog` and machine learning frameworks is the [Tensor](https://github.com/kevbuh/froog/blob/977b09caf32f21904768b08b2772139596604603/froog/tensor.py#L47). A [tensor](https://en.wikipedia.org/wiki/Tensor_(machine_learning)) is simply a matrix of matrices (more accurately a multi-dimensional array). 

You can create a Tensor in `froog` with:
```python
import numpy as np
from froog.tensor import Tensor
my_tensor = Tensor([1,2,3])
```

# Creating a model

Okay cool, so now you know that `froog`'s main datatype is a Tensor and uses NumPy in the background. How do I actually build a model? 

Here's an example of how to create an MNIST multi-layer perceptron (MLP). We wanted to make it as simple as possible for you to do so it resembles very basic Python concepts like classes. There are really only two methods you need to define: 
1. `__init__` that defines layers of the model (here we use `Linear`) 
2. `forward` which defines how the input should flow through your model. We use a simple dot product with a `Linear` layer with a [`ReLU`](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation.

To create an instance of the `mnistMLP` model, do the same as you would in Python: `model = mnistMLP()`. 

We support a few different optimizers, [here](https://github.com/kevbuh/froog/blob/main/froog/optim.py) which include:
- [Stochastic Gradient Descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
- [Adaptive Moment Estimation (Adam)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam)
- [Root Mean Square Propagation (RMSProp)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp)

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

So there are two quick examples to get you up and running. You might have noticed some operations like `reshape` and were wondering what else you can do with `froog`. We have many more operations that you can apply on tensors: 

- `def __init__(self, data)`:

  - Tensor takes in one param, which is the data. Since `froog` has a NumPy backend, the input data into tensors has to be a NumPy array.
  - Tensor has a `self.data` state that it holds. this contains the data inside of the tensor.
  - In addition, it has `self.grad`. this is to hold what the gradients of the tensor is. 
  - Lastly, it has `self._ctx`. These are the internal variables used for autograd graph construction. This is where the backward gradient computations are saved. 

*Properties*

- `shape(self)`: this returns the tensor shape

*Methods*
- `def zeros(*shape)`: this returns a tensor full of zeros with any shape that you pass in. Defaults to np.float32
- `def ones(*shape)`: this returns a tensor full of ones with any shape that you pass in. Defaults to np.float32
- `def randn(*shape):`: this returns a randomly initialized Tensor of *shape

*Gradient calculations*

- `froog` computes gradients automatically through a process called automatic differentiation. it has a variable `_ctx`, which stores the chain of operations. It will take the current operation, let's say a dot product, and go to the dot product definition in `froog/ops.py`, which contains a backward pass specifically for dot products. all methods, from add to 2x2 maxpools, have this backward pass implemented.

*Functions*

The other base class in froog is the class `Function`. It keeps track of input tensors and tensors that need to be saved for backward passes

- `def __init__(self, *tensors)`: takes in an argument of tensors, which are then saved. 
- `def save_for_backward(self, *x)`: saves Tensors that are necessary to compute for the computation of gradients in the backward pass. 
- `def apply(self, arg, *x)`: takes care of the forward pass, applying the operation to the inputs.

*Register*

- `def register(name, fxn)`: allows you to add a method to a Tensor. This allows you to chain any operations, e.g. x.dot(w).relu(), where w is a tensor

# Environment Variables

| Variable | Purpose |
|----------|---------|
| WARNING=1 | Display warnings when tensor data isn't float32 (needed for numerical jacobian) |
| DEBUG=1 | Allow repeated warnings (don't suppress duplicates) |
| GPU=1 | Enable GPU acceleration via OpenCL |
| VIZ=1 | Enable visualization in EfficientNet model |
| CI=1 | Disable progress bars in tests for CI environments |

Multiple variables can be used together: `WARNING=1 DEBUG=1 GPU=1 python your_script.py` 

# Contributing

Theres lots of work to be done!

Here are some basic guidelines for contributing:
1. increase simplicity
2. increase efficiency
3. increase functionality, must include [tests](https://github.com/kevbuh/froog/tree/main/tests)

#### Small   <!-- ez money  -->
- ensemble trees 
- support vector machines
- basic linear regression model
- improve docs
- binary cross entropy
- flatten
- dropout 
#### Medium  <!-- mid tier -->
- faster conv
- simplify how context and gradients are handled
#### Large <!-- EXPERT LEVEL!!!  -->
- float16 support
- transformers
- stable diffusion
- winograd convs
- GPU Support
  - CUDA
  - AMD

### Tests
Tests are located [here](https://github.com/kevbuh/froog/tree/main/tests).

You can run them in your terminal by going into the root folder and entering:

```bash
python -m pytest
```