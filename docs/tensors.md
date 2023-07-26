# Tensors

Tensors are the fundamental datatype in froog, and one of the two main classes.

```def __init__(self, data)```:

Tensor takes in one param, which is the data. Since froog has a numpy backend, the input data into tensors has to be a numpy array.

Tensor has a ```self.data``` state that it holds. this contains the data inside of the tensor.

In addition, it has ```self.grad```. this is to hold what the gradients of the tensor is. 

Lastly, it has ```self._ctx```. theser are the internal vairables used for autograd graph construction. put more simply, this is where the backward gradient computations are saved. 

### Properties

```shape(self)```: this returns the tensor shape


### methods
```def zeros(*shape)```: this returns a tensor full of zeros with any shape that you pass in. Defaults to np.float32

```def ones(*shape)```: this returns a tensor full of ones with any shape that you pass in. Defaults to np.float32

```def randn(*shape):```: this returns a randomly initialized Tensor of *shape

### Backward pass
Backpropogation is the way in which neural networks learn. By using the chain rule from calculus, you can go backwards per operation and compute how much that weight affected the models output.

froog computes gradients automatically through a process called automatic differentiation. it has a variable ```_ctx```, which stores the chain of operations. it will take the current operation, lets say a dot product, and go to the dot product definition in ```froog/ops.py```, which contains a backward pass specfically for dot products. all methods, from add to 2x2 maxpools, have this backward pass implemented.

# Functions

The other base class in froog is the class Function. It keeps track of input tensors and tensors that need to be saved for backward passes

```def __init__(self, *tensors)```: takes in an argument of tensors, which are then saved. 

```def save_for_backward(self, *x)```: saves Tensors that are necessary to compute for the computation of gradients in the backward pass. 

```def apply(self, arg, *x)```: This is what makes everything work. The apply() method takes care of the forward pass, applying the operation to the inputs.

# Register

```def register(name, fxn)```: this function allows you to add a method to a Tensor. This allows you to chain any operations, e.g. x.dot(w).relu(), where w is a tensor