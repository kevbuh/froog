# froog <img src="https://github.com/kevbuh/froog/actions/workflows/test.yml/badge.svg" alt="unit test badge" > <img src="https://static.pepy.tech/badge/froog" alt="num downloads badge">
<div align="center" >
  <img src="https://raw.githubusercontent.com/kevbuh/froog/main/assets/froog.png" alt="froog the frog" height="200">
  <br/>
  froog: a gpu accelerated tensor library
  <br/>
  <a href="https://github.com/kevbuh/froog">homepage</a> | <a href="https://github.com/kevbuh/froog/tree/main/DOCS.md">documentation</a> | <a href="https://pypi.org/project/froog/">pip</a>
  <br/>
  <br/>
</div>

```froog``` is an easy-to-read tensor library (<a href="https://www.pepy.tech/projects/froog">27k pip installs!</a>) with support for GPU acceleration with [OpenCL](https://www.khronos.org/opencl/) and [Apple Metal](https://developer.apple.com/metal/). Inspired by [tinygrad](https://github.com/tinygrad/tinygrad), and [micrograd](https://github.com/karpathy/micrograd).

## Installation
```bash
pip install froog
```

## Features
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
- <a href="https://github.com/kevbuh/froog/blob/main/froog/gpu">GPU Support</a> 

## Quick Example

Here's how you set up a simple multilayer perceptron for classification on MNIST. Looks pretty similar to pytorch, right?

```python
from froog.tensor import Tensor
from froog.nn import Linear
import froog.optim as optim

class mnistMLP:
  def __init__(self):
    self.l1 = Tensor(Linear(784, 128)) # layer 1
    self.l2 = Tensor(Linear(128, 10))  # layer 2

  def forward(self, x):
    # forward pass through both layers and softmax for output probabilities
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax() 

model = mnistMLP() # create model
optim = optim.SGD([model.l1, model.l2], lr=0.001) # stochastic gradient descent optimizer
```

## GPU Support

Device management is handled transparently and will automatically select one of ```[METAL, OPENCL, CPU]```. To use the GPU:

```python
from froog.tensor import Tensor
from froog import get_device
# Check if GPU is available
has_gpu = get_device() is not None and get_device().name != "CPU"
# Create a tensor
x = Tensor([1, 2, 3])
# Push to GPU if available
if has_gpu: x = x.to_gpu()
# Operations run on GPU automatically
y = x + x
z = y * y
# Bring back to CPU when needed
result = z.to_cpu()
print(result.data)
```

You can also check what devices are available:

```python
from froog import get_available_devices
available_devices = get_available_devices()
print(f"Available devices: {available_devices}")
```

Or set a specific device:

```python
from froog import set_device
set_device("METAL")  # or "OPENCL"
```

## EfficientNet in froog!

<img src="https://github.com/kevbuh/froog/blob/main/assets/efficientnet_pug.png" alt="pug" height="200">

We have an implementation of [EfficientNet v2](https://arxiv.org/abs/2104.00298) built entirely in ```froog``` using the official PyTorch weights! Running inference on this pug...


```bash
python3 models/efficientnet.py <https://optional_image_url>

***********output*************
inference 4.34 s

imagenet class: 254
prediction    : pug, pug-dog
probability   : 0.9402361
******************************
```

I would recommend checking out the <a href="https://github.com/kevbuh/froog/blob/main/models/efficientnet.py">code</a>, it's highly documented and pretty cool.

<!-- ## Contributing -->
<!-- THERES LOT OF STUFF TO WORK ON! VISIT THE <a href="https://github.com/kevbuh/froog/blob/main/docs/bounties.md">BOUNTY SHOP</a>  -->

<!-- Pull requests will be merged if they:
* increase simplicity
* increase functionality
* increase efficiency

More info on <a href="https://github.com/kevbuh/froog/blob/main/docs/contributing.md">contributing</a>. Make sure to run ```python -m pytest``` before creating a PR. -->

## API

MATH
- ```.add(y)``` - Addition with y
- ```.sub(y)``` - Subtraction with y
- ```.mul(y)``` - Multiplication with y
- ```.div(y)``` - Division by y
- ```.pow(y)``` - Power function (raise to power y)
- ```.sum()``` - Sum all elements
- ```.mean()``` - Mean of all elements
- ```.sqrt()``` - Square root
- ```.dot(y)``` - Matrix multiplication with y
- ```.matmul(y)``` - Alias for dot

MACHINE LEARNING
- ```.relu()``` - Rectified Linear Unit activation
- ```.sigmoid()``` - Sigmoid activation
- ```.dropout(p=0.5, training=True)``` - Dropout regularization
- ```.logsoftmax()``` - Log softmax function
- ```.swish()``` - Swish activation function (x * sigmoid(x))
- ```.conv2d(w, stride=1, groups=1)``` - 2D convolution
- ```.im2col2dconv(w)``` - Image to column for convolution
- ```.max_pool2d(kernel_size=(2,2))``` - 2D max pooling
- ```.avg_pool2d(kernel_size=(2,2))``` - 2D average pooling

TENSOR
- ```Tensor.zeros(*shape)``` - Create tensor of zeros
- ```Tensor.ones(*shape)``` - Create tensor of ones
- ```Tensor.randn(*shape)``` - Create tensor with random normal values
- ```Tensor.eye(dim)``` - Create identity matrix
- ```Tensor.arange(start, stop=None, step=1)``` - Create tensor with evenly spaced values

TENSOR PROPERTIES
- ```.shape``` - The shape of the tensor as a tuple
- ```.size``` - Total number of elements in the tensor
- ```.ndim``` - Number of dimensions (rank) of the tensor
- ```.transpose``` - Transpose of the tensor
- ```.dtype``` - Data type of the tensor
- ```.is_gpu``` - Whether tensor is on GPU
- ```.grad``` - Gradient of tensor with respect to some scalar value
- ```.data``` - Underlying NumPy array (or GPU buffer)
- ```.to_float()``` - Converts tensor to float32 data type
- ```.to_int()``` - Converts tensor to int32 data type
- ```.to_bool()``` - Converts tensor to boolean data type
- ```.reshape(*shape)``` - Change tensor shape
- ```.view(*shape)``` - Alternative to reshape
- ```.pad2d(padding=None)``` - Pad 2D tensors
- ```.flatten()``` - Returns a flattened 1D copy of the tensor
- ```.unsqueeze(dim)``` - Add dimension of size 1 at specified position
- ```.squeeze(dim=None)``` - Remove dimensions of size 1
- ```.detach()``` - Returns a tensor detached from computation graph
- ```.assign(x)``` - Assign values from tensor x to this tensor

GPU
- ```.to_cpu()``` - Moves tensor to CPU
- ```.to_gpu()``` - Moves tensor to GPU 
- ```.gpu_()``` - In-place GPU conversion (modifies tensor)

AUTOGRAD
- ```.backward(allow_fill=True)``` - Performs backpropagation
