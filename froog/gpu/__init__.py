from froog.gpu.device import Device
from froog.gpu.cl.device_cl import OpenCLDevice
from froog.gpu.cl.device_cl import CL_AVAILABLE

# Try to import Metal
try:
    from froog.gpu.metal.device_metal import MetalDevice
    from froog.gpu.metal.metal_utils import METAL_AVAILABLE
except ImportError:
    METAL_AVAILABLE = False

# Make device classes available at package level
__all__ = ['Device', 'OpenCLDevice', 'MetalDevice', 'get_device', 'set_device']

# Global device instance
_current_device = None

def get_device() -> Device:
    """
    Get the current active device.
    If no device is set, it attempts to initialize a device in this order:
    1. Metal (on macOS)
    2. OpenCL (cross-platform)
    """
    global _current_device
    if _current_device is None:
        # Try Metal first (on macOS)
        if 'METAL_AVAILABLE' in globals() and METAL_AVAILABLE:
            try:
                _current_device = MetalDevice()
                # Check if Metal device is actually available
                if hasattr(_current_device, "device") and _current_device.device is not None:
                    return _current_device
                _current_device = None
            except Exception as e:
                import os
                if os.getenv("DEBUG") == "1":
                    print(f"Metal initialization failed: {e}")
                _current_device = None
        
        # Fall back to OpenCL
        if CL_AVAILABLE:
            try:
                _current_device = OpenCLDevice()
                if not _current_device.is_available():
                    _current_device = None
                    import os
                    if os.getenv("DEBUG") == "1":
                        print("OpenCL device not available")
            except ImportError as e:
                import os
                if os.getenv("DEBUG") == "1":
                    print(f"OpenCL import error: {e}")
                _current_device = None
            except Exception as e:
                import os
                if os.getenv("DEBUG") == "1":
                    print(f"OpenCL initialization error: {e}")
                _current_device = None
    
    return _current_device

def set_device(device: Device) -> None:
    """Set the active device."""
    global _current_device
    _current_device = device 