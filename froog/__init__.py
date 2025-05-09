import froog.optim
import froog.tensor
import froog.utils
import froog.gpu.cl.cl_utils

# Try to import Metal utils if available
try:
    import froog.gpu.metal.metal_utils
except ImportError:
    pass

# Import device management functions
from froog.gpu import Device, OpenCLDevice, get_device, set_device

# Try to import Metal device if available
try:
    from froog.gpu.metal import MetalDevice
    __all__ = ['Device', 'OpenCLDevice', 'MetalDevice', 'get_device', 'set_device']
except ImportError:
    __all__ = ['Device', 'OpenCLDevice', 'get_device', 'set_device']