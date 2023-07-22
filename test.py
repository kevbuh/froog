import numpy as np
from frog.tensor import Tensor

my_tensor = Tensor(np.array([1,2,3]))
my_tensor2 = Tensor(np.array([4,5,6]))
my_tensor3 = my_tensor.add(my_tensor2).mul(my_tensor2)

# print(my_tensor)