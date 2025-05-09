"""
Utility functions for working with GPU buffers.
This module provides functions for extracting data from GPU buffers and performing operations on them.
"""

import numpy as np
from typing import Any, Union, Tuple

def is_gpu_buffer(data: Any) -> bool:
    """
    Check if data is a GPU buffer (Metal or OpenCL).
    
    Args:
        data: The data to check
        
    Returns:
        bool: True if data is a GPU buffer, False otherwise
    """
    # For Metal buffers check for __pyobjc_object__ attribute or Buffer in type name
    if hasattr(data, "__pyobjc_object__") or str(type(data)).find('Buffer') >= 0:
        return True
    
    # For Metal buffers, check if it has a length method
    if hasattr(data, "length") and callable(getattr(data, "length")):
        return True
    
    return False

def get_buffer_data(buffer: Any) -> np.ndarray:
    """
    Extracts data from a buffer object, handling both numpy arrays and GPU buffers.
    For GPU buffers, tries to use the device metadata.
    
    Args:
        buffer: The buffer to extract data from
        
    Returns:
        np.ndarray: The extracted data as a NumPy array
    """
    # If it's already a numpy array, return it directly
    if isinstance(buffer, np.ndarray):
        return buffer
        
    # Import froog.gpu inside the function to avoid circular imports
    from froog.gpu import get_device
    
    # Check if it's a GPU buffer
    if is_gpu_buffer(buffer):
        # Try to get metadata from device
        device = get_device()
        if device and hasattr(device, 'buffer_metadata'):
            buffer_id = id(buffer)
            if buffer_id in device.buffer_metadata:
                # If we have a CPU copy in metadata, use that
                np_array = device.buffer_metadata[buffer_id].get('numpy_array')
                if np_array is not None:
                    return np_array
        
        # If no metadata or no cached NumPy array, download it from device
        if device:
            try:
                result = device.download_tensor(buffer)
                # Ensure the result is a numpy array, not another buffer
                if is_gpu_buffer(result):
                    print(f"Warning: Device returned a buffer instead of numpy array from {buffer}")
                    return np.zeros((1,), dtype=np.float32)
                return result
            except Exception as e:
                print(f"Error downloading buffer: {e}")
                return np.zeros((1,), dtype=np.float32)
        
        # If we got here, we have a GPU buffer but no device to download from
        print(f"Warning: GPU buffer detected but no device available to download data")
        return np.zeros((1,), dtype=np.float32)
    
    # For unknown types, return a default array
    print(f"Warning: Unknown buffer type {type(buffer)}, returning zeros")
    return np.zeros((1,), dtype=np.float32)

def buffer_add(x: Any, y: Any) -> np.ndarray:
    """Add two buffers or arrays"""
    return get_buffer_data(x) + get_buffer_data(y)

def buffer_sub(x: Any, y: Any) -> np.ndarray:
    """Subtract two buffers or arrays"""
    return get_buffer_data(x) - get_buffer_data(y)
    
def buffer_mul(x: Any, y: Any) -> np.ndarray:
    """Multiply two buffers or arrays"""
    return get_buffer_data(x) * get_buffer_data(y)
    
def buffer_div(x: Any, y: Any) -> np.ndarray:
    """Divide two buffers or arrays"""
    return get_buffer_data(x) / get_buffer_data(y)
    
def buffer_pow(x: Any, y: Any) -> np.ndarray:
    """Power operation on two buffers or arrays"""
    return get_buffer_data(x) ** get_buffer_data(y)
    
def buffer_dot(x: Any, y: Any) -> np.ndarray:
    """Matrix multiplication of two buffers or arrays"""
    return np.matmul(get_buffer_data(x), get_buffer_data(y))
    
def buffer_sum(x: Any) -> np.ndarray:
    """Sum all elements in a buffer or array"""
    return np.array([np.sum(get_buffer_data(x))])
    
def buffer_relu(x: Any) -> np.ndarray:
    """Apply ReLU to a buffer or array"""
    return np.maximum(get_buffer_data(x), 0)
    
def buffer_reshape(x: Any, shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape a buffer or array"""
    return get_buffer_data(x).reshape(shape)

def buffer_logsoftmax(x: Any) -> np.ndarray:
    """Apply log softmax to a buffer or array"""
    data = get_buffer_data(x)
    # Subtract max for numerical stability
    max_vals = np.max(data, axis=-1, keepdims=True)
    exp_vals = np.exp(data - max_vals)
    sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)
    return data - max_vals - np.log(sum_exp)

def buffer_pad2d(x: Any, padding: Tuple[int, int, int, int]) -> np.ndarray:
    """Pad a 4D tensor (N, C, H, W) with padding (left, right, top, bottom)"""
    data = get_buffer_data(x)
    # Check if we have a 4D tensor
    if len(data.shape) != 4:
        print(f"Warning: pad2d expects 4D tensor, got shape {data.shape}")
        return data
    
    # Extract padding values
    pad_left, pad_right, pad_top, pad_bottom = padding
    
    # Create padding config for np.pad
    # Format is ((before_1, after_1), (before_2, after_2), ...)
    pad_width = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    
    return np.pad(data, pad_width, mode='constant') 