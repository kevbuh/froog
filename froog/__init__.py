import froog.optim
import froog.tensor
import froog.utils
import froog.ops  # ensure tensor operations are registered on import

# Import GPU packages
import froog.gpu.cl.cl_utils

# Try to import Metal utils if available
try:
    import froog.gpu.metal.metal_utils
except ImportError:
    pass

# Import device management functions
from froog.gpu import (
    Device, OpenCLDevice, get_device, set_device,
    upload_tensor, download_tensor, is_buffer,
    allocate_buffer, synchronize, get_available_devices
)

# Try to import Metal device if available
try:
    from froog.gpu.metal import MetalDevice
    __all__ = [
        'Device', 'OpenCLDevice', 'MetalDevice', 
        'get_device', 'set_device', 'upload_tensor', 
        'download_tensor', 'is_buffer', 
        'allocate_buffer', 'synchronize', 'get_available_devices'
    ]
except ImportError:
    __all__ = [
        'Device', 'OpenCLDevice', 
        'get_device', 'set_device', 'upload_tensor', 
        'download_tensor', 'is_buffer', 
        'allocate_buffer', 'synchronize', 'get_available_devices'
    ]