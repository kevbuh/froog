# Standard library imports
import platform
import traceback

# Local imports
from froog.gpu.device import Device
from froog.gpu.cl.device_cl import OpenCLDevice, CL_AVAILABLE
from froog.gpu.metal.device_metal import MetalDevice
from froog.gpu.metal.metal_utils import METAL_AVAILABLE, check_and_initialize_metal

# Optional imports
try:
    from froog.gpu import buffer_utils
except ImportError:
    buffer_utils = None

# Make device classes available at package level
__all__ = ['Device', 'OpenCLDevice', 'MetalDevice', 'get_device', 'set_device', 'buffer_utils']

# Global device instance
_current_device = None
_device_info_printed = False


def get_device() -> Device:
    global _current_device, _device_info_printed
    if _current_device is None:
        if platform.system() == "Darwin":
            if check_and_initialize_metal(get_device, set_device):
                try:
                    metal_device = MetalDevice()
                    if metal_device.device is not None:
                        _current_device = metal_device
                        _device_info_printed = True
                        return _current_device
                except Exception:
                    traceback.print_exc()
        elif CL_AVAILABLE:
            try:
                _current_device = OpenCLDevice()
                if _current_device.is_available():
                    print(f"Using OpenCL device: {_current_device.get_capabilities()['name']}")
                    _device_info_printed = True
                else:
                    _current_device = None
                    print("OpenCL device not available")
            except ImportError as e:
                print(f"OpenCL import error: {e}")
                _current_device = None
            except Exception as e:
                print(f"OpenCL initialization error: {e}")
                _current_device = None

    if _current_device is None:
        print("No GPU device available. Using CPU fallback.")
    elif not _device_info_printed:
        device_type = "Metal" if isinstance(_current_device, MetalDevice) else "OpenCL" if isinstance(_current_device, OpenCLDevice) else "Unknown"
        try:
            device_name = _current_device.get_capabilities().get('name', 'unnamed')
            print(f"Using {device_type} device: {device_name}")
        except:
            print(f"Using {device_type} device")
        _device_info_printed = True

    return _current_device

def set_device(device: Device) -> None:
    """Set the active device."""
    global _current_device, _device_info_printed
    _current_device = device
    _device_info_printed = False  # Reset so we print info for the new device 