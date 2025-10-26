"""
Utility functions for working with GPU buffers.
This module provides functions for extracting data from GPU buffers and performing operations on them.
"""

import numpy as np
from typing import Any, Tuple
from froog.gpu import get_device

def is_gpu_buffer(data: Any) -> bool:
    """
    Check if data is a GPU buffer (Metal or OpenCL).
    Args: data: The data to check
    Returns: bool: True if data is a GPU buffer, False otherwise
    """
    if hasattr(data, "__pyobjc_object__") or 'Buffer' in str(type(data)): return True
    if hasattr(data, "length") and callable(getattr(data, "length")): return True
    return False

def get_buffer_data(buffer: Any) -> np.ndarray:
    """
    Extracts data from a buffer object, handling both numpy arrays and GPU buffers.
    For GPU buffers, tries to use the device metadata.
    Args: buffer: The buffer to extract data from
    Returns: np.ndarray: The extracted data as a NumPy array
    """
    if isinstance(buffer, np.ndarray): return buffer
    if is_gpu_buffer(buffer):
        device = get_device()
        if device and hasattr(device, 'buffer_metadata'):
            buffer_id = id(buffer)
            if buffer_id in device.buffer_metadata:
                np_array = device.buffer_metadata[buffer_id].get('numpy_array')
                if np_array is not None: return np_array

        if device:
            try:
                result = device.download_tensor(buffer)
                if is_gpu_buffer(result):
                    print(f"Warning: Device returned a buffer instead of numpy array from {buffer}")
                    return np.zeros((1,), dtype=np.float32)
                return result
            except Exception as e:
                print(f"Error downloading buffer: {e}")
                return np.zeros((1,), dtype=np.float32)

        print("Warning: GPU buffer detected but no device available to download data")
        return np.zeros((1,), dtype=np.float32)

    print(f"Warning: Unknown buffer type {type(buffer)}, returning zeros")
    return np.zeros((1,), dtype=np.float32)

def buffer_add(x: Any, y: Any) -> np.ndarray: return get_buffer_data(x) + get_buffer_data(y)
def buffer_sub(x: Any, y: Any) -> np.ndarray: return get_buffer_data(x) - get_buffer_data(y)
def buffer_mul(x: Any, y: Any) -> np.ndarray: return get_buffer_data(x) * get_buffer_data(y)
def buffer_div(x: Any, y: Any) -> np.ndarray: return get_buffer_data(x) / get_buffer_data(y)
def buffer_pow(x: Any, y: Any) -> np.ndarray: return get_buffer_data(x) ** get_buffer_data(y)
def buffer_dot(x: Any, y: Any) -> np.ndarray: return np.matmul(get_buffer_data(x), get_buffer_data(y))
def buffer_sum(x: Any) -> np.ndarray: return np.array([np.sum(get_buffer_data(x))])
def buffer_relu(x: Any) -> np.ndarray: return np.maximum(get_buffer_data(x), 0)
def buffer_reshape(x: Any, shape: Tuple[int, ...]) -> np.ndarray: return get_buffer_data(x).reshape(shape)

def buffer_logsoftmax(x: Any) -> np.ndarray:
    data = get_buffer_data(x)
    max_vals = np.max(data, axis=-1, keepdims=True)
    exp_vals = np.exp(data - max_vals)
    sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)
    return data - max_vals - np.log(sum_exp)

def buffer_pad2d(x: Any, padding: Tuple[int, int, int, int]) -> np.ndarray:
    data = get_buffer_data(x)
    if len(data.shape) != 4:
        print(f"Warning: pad2d expects 4D tensor, got shape {data.shape}")
        return data
    pad_left, pad_right, pad_top, pad_bottom = padding
    pad_width = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    return np.pad(data, pad_width, mode='constant') 
