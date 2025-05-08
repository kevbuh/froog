from froog.gpu.device import Device
from froog.gpu.cl.device_cl import OpenCLDevice, CL_AVAILABLE
try:
    from froog.gpu.metal.device_metal import MetalDevice, METAL_AVAILABLE
except ImportError:
    METAL_AVAILABLE = False

# Make device classes available at package level
__all__ = ['Device', 'OpenCLDevice', 'MetalDevice', 'get_device', 'set_device']

# Global device instance
_current_device = None

def get_device() -> Device:
    """
    Get the current active device.
    If no device is set, it defaults to Metal device on Apple platforms if available, 
    otherwise falls back to OpenCL device if available.
    """
    global _current_device
    if _current_device is None:
        # Try Metal first on Apple platforms
        if METAL_AVAILABLE:
            try:
                _current_device = MetalDevice()
                if not _current_device.is_available():
                    _current_device = None
            except ImportError:
                _current_device = None
        
        # Fall back to OpenCL if Metal is not available
        if _current_device is None and CL_AVAILABLE:
            try:
                _current_device = OpenCLDevice()
                if not _current_device.is_available():
                    _current_device = None
            except ImportError:
                _current_device = None
    
    return _current_device

def set_device(device: Device) -> None:
    """Set the active device."""
    global _current_device
    _current_device = device 