import numpy as np
from typing import Any, Tuple, Optional, Union, TypeVar, cast, List, Dict, Callable
import functools

# Check if Metal is available
try:
    import platform
    import Metal
    import objc
    
    # Verify we're on macOS
    if platform.system() != "Darwin":
        print("Metal is only supported on macOS")
        METAL_AVAILABLE = False
    else:
        # Try to create a device to verify Metal is actually working
        # print("Checking Metal availability...")
        test_device = Metal.MTLCreateSystemDefaultDevice()
        if test_device is not None:
            # print(f"Metal device detected: {test_device.name()}")
            METAL_AVAILABLE = True
            METAL_GPU = True
            # We don't need to manually release test_device in PyObjC
            test_device = None
        else:
            # print("Metal API available but no compatible GPU device found")
            METAL_AVAILABLE = False
            METAL_GPU = False
except ImportError as e:
    print(f"Metal not available: {str(e)}")
    METAL_AVAILABLE = False
    METAL_GPU = False
    Metal = None
    objc = None

# Lazily initialized MetalDevice instance
_metal_device = None

# GPU availability flag for Metal
METAL_GPU = METAL_GPU

def check_and_initialize_metal(get_device_func, set_device_func) -> bool:
    """Check and initialize Metal device if available."""
    global METAL_AVAILABLE, METAL_GPU, _metal_device
    
    if not METAL_AVAILABLE:
        try:
            import platform
            import Metal
            import objc
            
            if platform.system() != "Darwin":
                print("Metal is only supported on macOS")
                return False
            
            test_device = Metal.MTLCreateSystemDefaultDevice()
            if test_device is not None:
                METAL_AVAILABLE = True
                METAL_GPU = True
                test_device = None
            else:
                return False
        except ImportError as e:
            print(f"Metal not available: {str(e)}")
            return False
    
    if _metal_device is None:
        from .device_metal import MetalDevice
        try:
            _metal_device = MetalDevice()
            if _metal_device.device is not None:
                print("using METAL")
                set_device_func(_metal_device)
                return True
            else:
                return False
        except Exception as e:
            print(f"Failed to initialize Metal device: {e}")
            return False
    
    return METAL_AVAILABLE and _metal_device is not None

def tensor_to_cpu(tensor: Any, get_device_func) -> np.ndarray:
    """Convert a GPU tensor to CPU"""
    device = get_device_func()
    if tensor.gpu:
        if device is None:
            raise Exception("No GPU device available")
        return device.download_tensor(tensor.data)
    else:
        return tensor.data

def tensor_to_gpu(data: np.ndarray, get_device_func) -> Any:
    """Convert CPU data to GPU buffer"""
    device = get_device_func()
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