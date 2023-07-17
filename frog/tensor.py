# inspired by pytorch
# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
# inspired by tinygrad

from functools import partialmethod
import numpy as np

# *********** Main Classes ***********
# ********* Tensor, Function *********
class Tensor:
    def __init__(self, data):
        if type(data) != np.ndarray:
            print(f"error constructing tensor with {data}")
            assert False
        self.data = data
        self.grad = None

        # internal variables used for autograd graph construction
        self._ctx = None

    def __str__(self):
        return f"Tensor {self.data} with grad {self.grad}" 

    def backward(self, allow_fill=True):
        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            # fill in the first grad with one
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)

        assert self.grad is not None
        
        # autograd engine
        grads = self._ctx.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t, g in zip(self._ctx.parents, grads):
            if g.shape != t.data.shape:
                print(
                    f"grad shape must match tensor shape in {self._ctx}, {g.shape} != {t.data.shape}"
                )
                assert False
            t.grad = g
            t.backward(False)

    def mean(self):
        div = Tensor(np.array([1 / self.data.size]))
        return self.sum().mul(div)

# Function class includes the context
class Function:
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []

    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

    # note that due to how partialmethod works, self and arg are switched
    # self is the tensor                   (a)
    # arg is the method                    (.dot, .relu) 
    # *x is b --> the input to the method  (a.dot(b), a.add(b))
    def apply(self, arg, *x):
        ctx = arg(self, *x)
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        ret._ctx = ctx
        return ret

# mechanism that allows you to chain methods in an intuitive and Pythonic way
# e.g. x.dot(w).relu(), where w is a tensor.
def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))

# *********** Elementary Functions ***********
# ********* Add, Mul, ReLU, Dot, Sum *********
# grad_output is the gradient of the loss with respect to the output of the operation.

class Add(Function):
    @staticmethod # @staticmethod doesn't require an instance of Add to work
    def forward(ctx, x, y):
        return x + y
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output 
register("add", Add)

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y * grad_output, x * grad_output
register("mul", Mul)


class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.maximum(input, 0)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.copy() # numpy only creates reference if you don't .copy()
        grad_input[input < 0] = 0
        return grad_input
register("relu", ReLU)


class Dot(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return input.dot(weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.T.dot(input).T
        return grad_input, grad_weight
register('dot', Dot)

# reduces its input tensor to a single value by summing all the elements
class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([input.sum()])

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output * np.ones_like(input)
register("sum", Sum)

# converts a vector of numbers into a vector of probabilities
# probabilities of each value are proportional to the scale of each value 
class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input):
        def logsumexp(x):
            c = x.max(axis=1)
            return c + np.log(np.exp(x - c.reshape((-1, 1))).sum(axis=1)) # axis=1 refers to the columns

        output = input - logsumexp(input).reshape((-1, 1))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        return grad_output - np.exp(output) * grad_output.sum(axis=1).reshape((-1, 1))
register("logsoftmax", LogSoftmax)
