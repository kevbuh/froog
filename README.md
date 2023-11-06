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

froog is a SUPER SIMPLE machine learning framework with the goal of creating tools with AI --> easily and efficiently.

froog encapsulates everything from <a href="https://github.com/kevbuh/froog/blob/main/models/linear_regression.py">linear regression</a> to <a href="https://github.com/kevbuh/froog/blob/main/models/efficientnet.py">convolutional neural networks </a>

all of this in under 1000 lines. 
<!-- in the <a href="https://github.com/kevbuh/froog/tree/main/tadpole">tadpole folder</a>. -->

# Installation
```bash
pip install froog
```

### Overview of Features
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

### Sneak Peek
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

# Bounties
THERES LOT OF STUFF TO WORK ON! VISIT THE <a href="https://github.com/kevbuh/froog/blob/main/docs/bounties.md">BOUNTY SHOP</a>

Pull requests will be merged if they:
* increase simplicity
* increase functionality
* increase efficiency

more info on <a href="https://github.com/kevbuh/froog/blob/main/docs/contributing.md">contributing</a>
