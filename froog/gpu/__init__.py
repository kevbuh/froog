from froog.gpu.device import Device
from froog.gpu.cl.device_cl import OpenCLDevice
from froog.gpu.cl.device_cl import CL_AVAILABLE

# Import buffer utilities
try:
    from froog.gpu import buffer_utils
except ImportError:
    pass

# Try to import Metal
try:
    from froog.gpu.metal.device_metal import MetalDevice
    from froog.gpu.metal.metal_utils import METAL_AVAILABLE
except ImportError:
    METAL_AVAILABLE = False

# Import the centralized Metal check function
from froog.gpu.metal.metal_utils import check_and_initialize_metal

# Make device classes available at package level
__all__ = ['Device', 'OpenCLDevice', 'MetalDevice', 'get_device', 'set_device', 'buffer_utils']

# Global device instance
_current_device = None
_device_info_printed = False

# Ensure MetalDevice is imported before it is used
from froog.gpu.metal.device_metal import MetalDevice

def get_device() -> Device:
    """
    Get the current active device.
    If no device is set, it attempts to initialize a device in this order:
    1. Metal (on macOS)
    2. OpenCL (cross-platform)
    """
    global _current_device, _device_info_printed
    if _current_device is None:
        import platform
        
        # Print system info for debugging
        # print(f"System: {platform.system()} {platform.release()}")
        # print(f"Python: {platform.python_version()}")
        
        # Try Metal first (on macOS)
        if platform.system() == "Darwin":
            # Use the centralized function to check and initialize Metal
            if check_and_initialize_metal(get_device, set_device):
                try:
                    metal_device = MetalDevice()
                    if metal_device.device is not None:
                        # print(f"Successfully initialized Metal device: {metal_device.device.name()}")
                        _current_device = metal_device
                        _device_info_printed = True
                        return _current_device
                except Exception as e:
                    import traceback
                    # print(f"Error initializing Metal device: {e}")
                    traceback.print_exc()
            # except ImportError as e:
                # print(f"Failed to import Metal modules: {e}")
                # print("Metal is not available on this Mac")
                # pass
        else:
            # print(f"Metal is not available on {platform.system()}")
            pass
        
        # Fall back to OpenCL
        if CL_AVAILABLE:
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
        # Print device info if we have a device but haven't printed info yet
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