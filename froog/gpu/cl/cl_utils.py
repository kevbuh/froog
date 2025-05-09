import functools
import numpy as np
from typing import Any, Tuple, Union

from .device_cl import OpenCLDevice
from .. import get_device

def init_gpu() -> None:
    """
    Ensures that a device is initialized and available
    """
    get_device()  # This will initialize a device if not already done

def tensor_to_cpu(tensor: Any) -> np.ndarray:
    """Convert a GPU tensor to CPU"""
    device = get_device()
    if tensor.gpu:
        if device is None: raise Exception("No GPU device available")
        return device.tensor_to_cpu(tensor.data)
    else:
        return tensor.data

def tensor_to_gpu(data: np.ndarray) -> Any:
    """Convert CPU data to GPU buffer"""
    device = get_device()
    if device is None: raise Exception("No GPU device available")
    return device.tensor_to_device(data)

def is_buffer(data: Any) -> bool:
    """Check if data is a GPU buffer"""
    device = get_device()
    if device is None: return False
    return device.is_buffer(data)

# Helper functions that delegate to the device instance

def get_size(x: Any) -> int:
    """Return the total number of elements in x"""
    device = get_device()
    if device is None: return int(np.prod(x.shape))
    return device.get_size(x)

def buffer_new(ctx: Any, shape: Tuple[int, ...]) -> Any:
    """Create a new empty GPU buffer with the given shape"""
    device = get_device()
    if device is None: raise Exception("No GPU device available")
    return device.buffer_new(shape)

def buffer_zeros(ctx: Any, shape: Tuple[int, ...]) -> Any:
    """Create a new GPU buffer filled with zeros"""
    device = get_device()
    if device is None: raise Exception("No GPU device available")
    return device.buffer_zeros(shape)

def buffer_like(ctx: Any, x: Any) -> Any:
    """Create a new GPU buffer with the same shape as x"""
    device = get_device()
    if device is None: raise Exception("No GPU device available")
    return device.buffer_like(x)

@functools.lru_cache
def clbuild(cl_ctx: Any, prg: str) -> Any:
    """Build an OpenCL program"""
    device = get_device()
    if device is None: return None
    return device.build_program(prg)

def binary_op(ctx: Any, code: str, x: Any, y: Any) -> Any:
    """Apply a binary operation to two GPU tensors"""
    device = get_device()
    if device is None:
        raise Exception("No GPU device available")
    return device.binary_op(code, x, y)

def unary_op(ctx: Any, code: str, x: Any) -> Any:
    """Apply a unary operation to a GPU tensor"""
    device = get_device()
    if device is None: raise Exception("No GPU device available")
    return device.unary_op(code, x)

@functools.lru_cache
def cl_pooling_krnl_build(cl_ctx: Any, iter_op: str, result_op: str, init_val: Union[int, str] = 0) -> Any:
    """Build an OpenCL kernel for pooling operations"""
    device = get_device()
    if device is None: return None
    if isinstance(device, OpenCLDevice): return device._build_pooling_kernel(iter_op, result_op, init_val)
    return None

def pooling_op(ctx: Any, input: Any, kernel_size: Tuple[int, int],  iter_op: str, result_op: str, init_val: Union[int, str] = 0) -> Any:
    """Apply a pooling operation to a GPU tensor"""
    device = get_device()
    if device is None: raise Exception("No GPU device available")
    ret = device.pooling_op(input, kernel_size, iter_op, result_op, init_val)
    ctx.data = np.empty((input.shape[0], input.shape[1], input.shape[2], input.shape[3])) # set shape expectation on tensor instance
    return ret 