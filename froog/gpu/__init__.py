from froog.gpu.device import Device
from froog.gpu.cl.device_cl import OpenCLDevice

# Make device classes available at package level
__all__ = ['Device', 'OpenCLDevice', 'get_device']

# Global device instance
_current_device = None

def get_device() -> Device:
    """
    Get the current active device.
    If no device is set, it defaults to OpenCL device if available.
    """
    global _current_device
    if _current_device is None:
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