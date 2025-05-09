from froog.gpu.device import Device
from froog.gpu.cl.device_cl import OpenCLDevice
from froog.gpu.device_manager import get_device, set_device, upload_tensor, download_tensor,  is_buffer, allocate_buffer, synchronize, get_available_devices, _DEVICE_MANAGER as DeviceManager
try:
    from froog.gpu.metal.device_metal import MetalDevice
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    MetalDevice = None
    METAL_AVAILABLE = False
try: from froog.gpu import buffer_utils
except ImportError: buffer_utils = None
__all__ = [ 'Device', 'OpenCLDevice', 'get_device', 'set_device',  'upload_tensor', 'download_tensor', 'is_buffer', 'allocate_buffer', 'synchronize', 'get_available_devices',  'buffer_utils' ]
if HAS_METAL: __all__.append('MetalDevice')
get_device() 