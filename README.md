# ribbit <img src="https://github.com/kevbuh/ribbit/actions/workflows/test.yml/badge.svg" alt="unit test badge" >
<div align="center" >
  <img src="https://raw.githubusercontent.com/kevbuh/ribbit/main/assets/froog.png" alt="froog the frog" height="200">
  <br/>
  ribbit: fast real-time optimization of gradients 
  <br/>
  a beautifully compact machine-learning library
  <br/>
  <a href="https://github.com/kevbuh/ribbit">homepage</a> | <a href="https://github.com/kevbuh/ribbit/tree/main/docs">documentation</a> | <a href="https://pypi.org/project/froog/">pip</a>
  <br/>
  <br/>
</div>

RIBBIT is a SUPER SIMPLE machine learning framework with the goal of creating tools with AI --> easily and efficiently.

RIBBIT encapsulates everything from <a href="https://github.com/kevbuh/ribbit/blob/main/models/linear_regression.py">linear regression</a> to <a href="https://github.com/kevbuh/ribbit/blob/main/models/efficientnet.py">convolutional neural networks </a>

# Installation
```bash
pip install froog
```

### Overview of Features
- <a href="https://github.com/kevbuh/ribbit/blob/main/ribbit/tensor.py">Custom Tensors</a> 
  - Backpropagation
  - Automatic Differentiation (autograd)
      - Forward and backward passes
- <a href="https://github.com/kevbuh/ribbit/blob/main/ribbit/ops.py">ML Operations</a> 
  - 2D Convolutions (im2col)
  - Numerical gradient checking
  - Acceleration methods (Adam)
  - Avg & Max pooling
- <a href="https://github.com/kevbuh/ribbit/blob/main/models/efficientnet.py">EfficientNet</a> inference
- <a href="https://github.com/kevbuh/ribbit/blob/main/ribbit/ops_gpu.py">GPU Support</a> 
- and a bunch <a href="https://github.com/kevbuh/ribbit/tree/main/ribbit">more</a> 

### Sneak Peek
```python
from ribbit.tensor import Tensor
from ribbit.nn import Linear
import ribbit.optim as optim

class mnistMLP:
  def __init__(self):
    self.l1 = Tensor(Linear(784, 128))
    self.l2 = Tensor(Linear(128, 10))

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = mnistMLP()
optim = optim.SGD([model.l1, model.l2], lr=0.001)
```

# Bounties
THERES LOT OF STUFF TO WORK ON! VISIT THE <a href="https://github.com/kevbuh/ribbit/blob/main/docs/bounties.md">BOUNTY SHOP</a>

Pull requests will be merged if they:
* increase simplicity
* increase functionality
* increase efficiency

more info on <a href="https://github.com/kevbuh/ribbit/blob/main/docs/contributing.md">contributing</a>