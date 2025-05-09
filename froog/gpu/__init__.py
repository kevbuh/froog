# Standard library imports
import platform

# Local imports
from froog.gpu.device import Device
from froog.gpu.cl.device_cl import OpenCLDevice, CL_AVAILABLE

# Import the device manager
from froog.gpu.device_manager import (
    get_device, set_device, upload_tensor, download_tensor, 
    is_buffer, allocate_buffer, synchronize, get_available_devices,
    _DEVICE_MANAGER as DeviceManager
)

# Try to import Metal if available
try:
    from froog.gpu.metal.device_metal import MetalDevice
    from froog.gpu.metal.metal_utils import METAL_AVAILABLE
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    MetalDevice = None
    METAL_AVAILABLE = False

# Optional imports
try:
    from froog.gpu import buffer_utils
except ImportError:
    buffer_utils = None

# Make device classes available at package level
__all__ = [
    'Device', 'OpenCLDevice', 'get_device', 'set_device', 
    'upload_tensor', 'download_tensor', 'is_buffer',
    'allocate_buffer', 'synchronize', 'get_available_devices', 
    'buffer_utils'
]

# Add Metal to __all__ if available
if HAS_METAL:
    __all__.append('MetalDevice')

# Initialize the device
get_device() 