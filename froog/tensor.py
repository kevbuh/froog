#  _______  ______    _______  _______  _______ 
# |       ||    _ |  |       ||       ||       |
# |    ___||   | ||  |   _   ||   _   ||    ___|
# |   |___ |   |_||_ |  | |  ||  | |  ||   | __ 
# |    ___||    __  ||  |_|  ||  |_|  ||   ||  |
# |   |    |   |  | ||       ||       ||   |_| |
# |___|    |___|  |_||_______||_______||_______|

import os
import numpy as np
from inspect import signature
from typing import Tuple, List, Union, Optional, Any, TypeVar, cast

from froog.gpu import (
    get_device, set_device, upload_tensor, download_tensor, 
    is_device_tensor, allocate_buffer, synchronize
)

# For backward compatibility
is_buffer = is_device_tensor
tensor_to_cpu = download_tensor
tensor_to_gpu = upload_tensor

# Function to initialize GPU
def init_gpu() -> None:
    get_device()

T = TypeVar('T', bound='Tensor')

class Tensor:
    did_float_warning = False

    def __init__(self, data: Union[List, np.ndarray, Any], gpu: bool = False):
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif is_buffer(data):
            self.gpu = True
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Error constructing tensor with {data}")
        if isinstance(data, np.ndarray):
            if data.dtype != np.float32 and not Tensor.did_float_warning:
                if os.getenv("WARNING") == "1":
                    print(f"warning, {data.shape} isn't float32. float64 needed for numerical jacobian")
                if not os.getenv("DEBUG") == "1":
                    Tensor.did_float_warning = True
            self.gpu = False
        self.data = data
        self.grad: Optional[Tensor] = None
        self._ctx = None
        if gpu:
            self.gpu_()

    def __repr__(self) -> str:
        return f"Tensor data: {self.data}, gradients: {self.grad.data if self.grad else None}"

    def assign(self, x: T) -> None:
        self.data = x.data

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.gpu:
            device = get_device()
            if device is not None and hasattr(device, 'buffer_metadata'):
                buffer_id = id(self.data)
                if buffer_id in device.buffer_metadata:
                    return device.buffer_metadata[buffer_id]['shape']
            try:
                data = tensor_to_cpu(self)
                return data.shape
            except Exception as e:
                print(f"Warning: Failed to get shape from GPU tensor: {e}")
                return (1,)
        return self.data.shape

    @property
    def size(self, dim=None) -> Union[int, Tuple[int, ...]]:
        if dim is not None:
            return self.shape[dim]
        return int(np.prod(self.shape))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def transpose(self) -> T:
        if isinstance(self.data, np.ndarray):
            return Tensor(self.data.T, gpu=self.gpu)
        else:
            cpu_tensor = self.to_cpu()
            return Tensor(cpu_tensor.data.T, gpu=self.gpu)

    @property
    def dtype(self) -> np.dtype:
        if self.gpu:
            device = get_device()
            if device is not None and hasattr(device, 'buffer_metadata'):
                buffer_id = id(self.data)
                if buffer_id in device.buffer_metadata:
                    return device.buffer_metadata[buffer_id]['dtype']
            return np.float32
        return self.data.dtype

    @property
    def is_gpu(self) -> bool:
        return self.gpu

    @staticmethod
    def zeros(*shape: int) -> T:
        return Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def ones(*shape: int) -> T:
        return Tensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def randn(*shape: int) -> T:
        return Tensor(np.random.randn(*shape).astype(np.float32))

    @staticmethod
    def eye(dim: int) -> T:
        return Tensor(np.eye(dim).astype(np.float32))

    @staticmethod
    def arange(start: Union[int, float], stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1) -> T:
        if stop is None:
            stop = start
            start = 0
        return Tensor(np.arange(start, stop, step, dtype=np.float32))

    def flatten(self) -> T:
        return Tensor(self.data.reshape(-1), gpu=self.gpu)

    def detach(self) -> T:
        return Tensor(self.data.copy(), gpu=self.gpu)

    def view(self, *shape: int) -> T:
        return Tensor(self.data.reshape(shape), gpu=self.gpu)

    def to_float(self) -> T:
        return Tensor(self.data.astype(np.float32), gpu=self.gpu)

    def to_int(self) -> T:
        return Tensor(self.data.astype(np.int32), gpu=self.gpu)

    def to_bool(self) -> T:
        return Tensor(self.data.astype(bool), gpu=self.gpu)

    def unsqueeze(self, dim: int) -> T:
        shape = list(self.shape)
        if dim < 0:
            dim = len(shape) + 1 + dim
        shape.insert(dim, 1)
        return Tensor(self.data.reshape(shape), gpu=self.gpu)

    def squeeze(self, dim: Optional[int] = None) -> T:
        if dim is None:
            return Tensor(self.data.squeeze(), gpu=self.gpu)
        else:
            shape = list(self.shape)
            if dim < 0:
                dim = len(shape) + dim
            if 0 <= dim < len(shape) and shape[dim] == 1:
                shape.pop(dim)
            return Tensor(self.data.reshape(shape), gpu=self.gpu)

    def backward(self, allow_fill: bool = True) -> None:
        if self._ctx is None:
            return
        if self.grad is None and allow_fill:
            assert self.shape == (1,)
            self.grad = Tensor(np.ones(self.shape, dtype=self.dtype), gpu=self.gpu)
        assert self.grad is not None
        grads = self._ctx.backward(self._ctx, self.grad.data)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t, g in zip(self._ctx.parents, grads):
            if g is None:
                continue
            t_shape = t.shape
            if is_buffer(g):
                device = get_device()
                if device is not None and hasattr(device, 'buffer_metadata'):
                    buffer_id = id(g)
                    if buffer_id in device.buffer_metadata:
                        g_shape = device.buffer_metadata[buffer_id]['shape']
                    else:
                        try:
                            g_cpu = tensor_to_cpu(g)
                            g_shape = g_cpu.shape
                        except:
                            print(f"Warning: Could not determine shape of gradient in {self._ctx}")
                            g_shape = t_shape
            else:
                g_shape = g.shape
            if g_shape != t_shape:
                print(f"grad shape must match tensor shape in {self._ctx}, {g_shape} != {t_shape}")
                assert False
            t.grad = Tensor(g)
            t.backward(allow_fill=False)

    def to_cpu(self) -> T:
        if self.gpu:
            data = tensor_to_cpu(self)
            ret = Tensor(data)
            if self.grad:
                ret.grad = self.grad.to_cpu()
            return ret
        else:
            return cast(T, self)

    ops = {}
    ops_gpu = {}

    def mean(self) -> T:
        div = Tensor(np.array([1 / self.size], dtype=np.float32), gpu=self.gpu)
        return self.sum().mul(div)

    def sqrt(self) -> T:
        root = Tensor(np.zeros(self.shape, dtype=np.float32) + 0.5, gpu=self.gpu)
        return self.pow(root)

    def div(self, y: T) -> T:
        root = Tensor(np.zeros(self.shape, dtype=np.float32) - 1, gpu=self.gpu)
        return self.mul(y.pow(root))

class Function:
    def __init__(self, *tensors: Tensor) -> None:
        self.parents = tensors
        self.saved_tensors: List[Any] = []

    def save_for_backward(self, *x: Any) -> None:
        self.saved_tensors.extend(x)

    def apply(self, *x: Any, **kwargs: Any) -> Tensor:
        op = self
        ctx = op(*x)
        params = signature(op.forward).parameters
        for p in params.values():
            if p.default is not p.empty:
                setattr(ctx, p.name, p.default)
        for k, v in kwargs.items():
            setattr(ctx, k, v)
        ret = Tensor(op.forward(ctx, *[t.data for t in x], **kwargs))
        ret._ctx = ctx
        return ret

def register(name: str, fxn: Any, gpu: bool = False) -> None:
    if gpu:
        setattr(Tensor, name, lambda self, *x, **kwargs: fxn.apply(fxn, self, *x, **kwargs))
        Tensor.ops_gpu[name] = fxn
    else:
        Tensor.ops[name] = fxn

    def dispatch(self: Tensor, *x: Any, **kwargs: Any) -> Tensor:
        try:
            op_func = (Tensor.ops_gpu if self.gpu else Tensor.ops)[name]
            return op_func.apply(op_func, self, *x, **kwargs)
        except Exception as e:
            print(f"Error in {name} operation: {e}")
            if os.getenv("DEBUG") == "1":
                print(f"  Self: {self}")
                for i, arg in enumerate(x):
                    print(f"  Arg {i}: {arg}")
                print(f"  Kwargs: {kwargs}")
            raise

    setattr(Tensor, name, dispatch)

    if name in ['add', 'sub', 'mul', 'div']:
        setattr(Tensor, "__%s__" % name, dispatch)
        setattr(Tensor, "__i%s__" % name, lambda self, x: self.assign(dispatch(self, x)))

import froog.ops

# Check for GPU availability using the device manager
if get_device() is not None and get_device().name != "CPU":
    # We have a GPU device
    device = get_device()
    print(f"Using {device.name}")
    
    # Import the device-specific operations based on device type
    if device.__class__.__name__ == "MetalDevice":
        try:
            import froog.gpu.metal.ops_metal
        except ImportError:
            pass
    elif device.__class__.__name__ == "OpenCLDevice":
        try:
            import froog.gpu.cl.ops_cl
            print("OpenCL operations imported successfully")
        except ImportError:
            print("Failed to import OpenCL operations")

def to_cpu(self) -> T:
    if self.gpu:
        data = tensor_to_cpu(self)
        ret = Tensor(data)
        if self.grad:
            ret.grad = self.grad.to_cpu()
        return ret
    else:
        return cast(T, self)

def gpu_(self) -> None:
    init_gpu()
    if not self.gpu and get_device() is not None and get_device().name != "CPU":
        self.data = tensor_to_gpu(self.data)
        self.gpu = True
        if self.grad:
            self.grad.gpu_()

def to_gpu(self) -> T:
    device = get_device()
    if device is None or device.name == "CPU":
        raise Exception("no gpu support! install pyopencl or use a Metal-compatible device")
    if not self.gpu:
        init_gpu()
        if device is None:
            raise Exception("No GPU device available")
        gpu_data = tensor_to_gpu(self.data)
        ret = Tensor(gpu_data)
        ret.gpu = True
        if self.grad:
            ret.grad = self.grad.to_gpu()
        return ret
    else:
        return cast(T, self)

Tensor.to_cpu = to_cpu
Tensor.gpu_ = gpu_
Tensor.to_gpu = to_gpu

def print_computation_graph(tensor: Tensor) -> None:
    visited = set()
    def traverse(t: Tensor, depth: int = 0):
        if t in visited:
            return
        visited.add(t)
        indent = '  ' * depth
        print(f"{indent}Tensor: {t}, Grad: {t.grad}")
        if t._ctx is not None:
            print(f"{indent}Operation: {t._ctx.__class__.__name__}")
            for parent in t._ctx.parents:
                traverse(parent, depth + 1)
    traverse(tensor)