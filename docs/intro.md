# Intro

FROG (Fast Real-time Optimization of Gradients) is an easy to read machine learning library. Its main goals is to provide the most common operations in machine learning in a super simple manner. Its meant for those who are looking to get into machine learning, and want to actually understand how everything works before it is ultra optimized (which all modern ml libraries do).

# Where to start?
The most fundamental item in all of FROG and machine learning is the Tensor. A tensor is simply a matrix of matrices (more accurately a multi-dimensional array). 

You can create a Tensor in FROG by
```python
import numpy as np
from frog.tensor import Tensor

my_tensor = Tensor(np.array([1,2,3]))
```

Notice that we had to import NumPy. FROG is built with NumPy, which allows for general matrix operations. If you want to create a Tensor manually make sure that it is a Numpy array!