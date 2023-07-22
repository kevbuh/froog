# Tensors

Tensors are the fundamental datatype in frog, and one of the two main classes.

def __init__(self, data):

it takes in one param, which is the data. Since frog has a numpy backend, the input data into tensors has to be a numpy array.

it also has a ```self.data `` state that it holds. this contains the data inside of the tensor.

in addition, it has ```self.grad ```. this is to hold what the gradients of the tensor is. 

lastly, it has ```self._ctx ```. theser are the internal vairables used for autograd graph construction. put more simply, this is where the backward gradient computations are saved. 

### Properties

```shape(self)``: this returns the tensor shape


### methods
```def zeros(*shape)```: this returns a tensor full of zeros with any shape that you pass in. defaults to np.float32

```def ones(*shape)```: this returns a tensor full of ones with any shape that you pass in. defaults to np.float32

```def randn(*shape):```: this returns a randomly 