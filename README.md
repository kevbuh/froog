# frog <img src="https://github.com/kevbuh/frog/actions/workflows/test.yml/badge.svg" alt="unit test badge" >
<div align="center" >
  <img src="https://github.com/kevbuh/frog/blob/main/assets/froog.jpeg" alt="froog the frog" height="300">
  
  <br/>
  frog: fast real-time optimization of gradients 
  <br/>
  An autograd & tensor machine learning library
  <br/>
  <br/>
</div>

No extra clutter. It just works.

### Creating an MNIST classifier

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

### Overview of Features
- Tensors
- Automatic Differentiation
    - Forward and Backward passes
- Input/Grad shape-tracking
- MNIST example
- Compares data against PyTorch to ensure correctness
- JIT 2D Convolutions

### Math Operations
- Scalar-Matrix Multiplication
- Dot Product
- Sum
- ReLU
- Log Softmax
- 2D Convolution

# TODO:
- Simplify
- Numerical Gradcheck
- Winograd Conv
- Stable Diffusion
- EfficientNet v2
- Transformers
