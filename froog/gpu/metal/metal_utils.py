import numpy as np
from typing import Any, Tuple, Optional, Union, TypeVar, cast, List, Dict, Callable
import functools

# Check if Metal is available
try:
    import Metal
    import objc
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    Metal = None
    objc = None

from .. import get_device, set_device

# Lazily initialized MetalDevice instance
_metal_device = None

# GPU availability flag for Metal
METAL_GPU = METAL_AVAILABLE

def init_metal() -> None:
    """Initialize the Metal device if available"""
    global _metal_device
    
    if METAL_AVAILABLE and _metal_device is None:
        # Lazy import to avoid circular imports
        from .device_metal import MetalDevice
        try:
            _metal_device = MetalDevice()
            set_device(_metal_device)
            return True
        except Exception as e:
            print(f"Failed to initialize Metal device: {e}")
    
    return False

def tensor_to_cpu(tensor: Any) -> np.ndarray:
    """Convert a GPU tensor to CPU"""
    device = get_device()
    if tensor.gpu:
        if device is None:
            raise Exception("No GPU device available")
        return device.download_tensor(tensor.data)
    else:
        return tensor.data

def tensor_to_gpu(data: np.ndarray) -> Any:
    """Convert CPU data to GPU buffer"""
    device = get_device()
    if device is None:
        raise Exception("No GPU device available")
    return device.upload_tensor(data)

def is_buffer(data: Any) -> bool:
    """Check if data is a GPU buffer"""
    device = get_device()
    if device is None:
        return False
    
    # For Metal, we need to check if it's a Metal buffer
    if device.__class__.__name__ == "MetalDevice":
        return hasattr(data, "length") and callable(data.length)
    return False 