import numpy as np
from typing import Any

# Check if Metal is available
try:
    import platform
    import Metal
    import objc

    if platform.system() != "Darwin":
        print("Metal is only supported on macOS")
        METAL_AVAILABLE = False
    else:
        test_device = Metal.MTLCreateSystemDefaultDevice()
        METAL_AVAILABLE = test_device is not None
except ImportError as e:
    if __name__ == "__main__":
        print(f"Metal not available: {str(e)}")
    METAL_AVAILABLE = False
    Metal = None
    objc = None

def is_metal_buffer(data: Any) -> bool:
    """Check if the data is a Metal buffer."""
    return (hasattr(data, "__pyobjc_object__") or 
            (hasattr(data, "length") and callable(data.length)) or
            str(type(data)).find('Metal.MTLBuffer') >= 0)

def get_buffer_data(buffer: Any) -> np.ndarray:
    """Get data from a Metal buffer."""
    if hasattr(buffer, "contents"):
        import ctypes
        buffer_length = buffer.length()
        float_count = buffer_length // 4  # Assuming float32
        
        contents = buffer.contents()
        if contents is not None:
            ptr = ctypes.cast(contents, ctypes.POINTER(ctypes.c_float))
            return np.ctypeslib.as_array(ptr, shape=(float_count,))
    
    # Return empty array if we couldn't get data
    return np.array([], dtype=np.float32)

def check_and_initialize_metal(get_device_func, set_device_func) -> bool:
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
            METAL_AVAILABLE = test_device is not None
            METAL_GPU = METAL_AVAILABLE
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
        except Exception as e:
            print(f"Failed to initialize Metal device: {e}")
            return False

    return METAL_AVAILABLE and _metal_device is not None

def tensor_to_cpu(tensor: Any, get_device_func) -> np.ndarray:
    device = get_device_func()
    if tensor.gpu:
        if device is None:
            raise Exception("No GPU device available")
        return device.download_tensor(tensor.data)
    return tensor.data

def tensor_to_gpu(data: np.ndarray, get_device_func) -> Any:
    device = get_device_func()
    if device is None:
        raise Exception("No GPU device available")
    return device.upload_tensor(data)

def is_buffer(data: Any) -> bool:
    device = get_device()
    if device is None:
        return False

    if device.__class__.__name__ == "MetalDevice":
        return hasattr(data, "length") and callable(data.length)
    return False 