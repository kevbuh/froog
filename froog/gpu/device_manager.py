"""
- `get_device()`: Get the current device or default device
- `set_device()`: Set the active device
- `get_available_devices()`: Get a list of all available devices
- `upload_tensor()`: Upload data to the current device
- `download_tensor()`: Download data from the current device
- `allocate_buffer()`: Allocate a buffer on the current device
- `is_buffer()`: Check if data is a device buffer
- `synchronize()`: Synchronize operations on the current device
"""

import functools
import platform
import os
import atexit
import traceback
from typing import Dict, Optional, List, Any, Union, Set, Tuple

import numpy as np

from froog.gpu.device import Device
from froog.gpu.cl.device_cl import OpenCLDevice, CL_AVAILABLE
try:
    from froog.gpu.metal.device_metal import MetalDevice
    from froog.gpu.metal.metal_utils import METAL_AVAILABLE
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

# Define a simple CPU fallback device
class CPUDevice(Device):
    """Simple CPU fallback device implementation."""
    
    def __init__(self):
        """Initialize a CPU device."""
        self.buffer_metadata = {}
        
    @property
    def name(self):
        """Return the name of the device."""
        return "CPU"
        
    def is_available(self):
        """CPU is always available."""
        return True
        
    def allocate_memory(self, size: int):
        """Allocate CPU memory."""
        return np.zeros(size // 4, dtype=np.float32)  # size is in bytes, each float32 is 4 bytes
        
    def free_memory(self, buffer):
        """Free CPU memory."""
        buffer_id = id(buffer)
        if buffer_id in self.buffer_metadata:
            del self.buffer_metadata[buffer_id]
            
    def upload_tensor(self, host_array) -> object:
        """Upload to CPU is a no-op, just return a copy."""
        data = np.asarray(host_array, dtype=np.float32).copy()
        buffer_id = id(data)
        self.buffer_metadata[buffer_id] = {
            'shape': data.shape,
            'dtype': data.dtype,
            'numpy_array': data
        }
        return data
        
    def download_tensor(self, buffer) -> object:
        """Download from CPU is a no-op, just return a copy."""
        # If it's a Tensor object, get the data
        if hasattr(buffer, "data"):
            buffer = buffer.data
        
        # If it's a numpy array, return it directly
        if isinstance(buffer, np.ndarray):
            return buffer.copy()
            
        # Try buffer metadata
        buffer_id = id(buffer)
        if buffer_id in self.buffer_metadata:
            if 'numpy_array' in self.buffer_metadata[buffer_id]:
                return self.buffer_metadata[buffer_id]['numpy_array'].copy()
        
        # Last resort
        try:
            return np.array(buffer, dtype=np.float32)
        except:
            print(f"Warning: Unable to convert buffer to numpy array: {type(buffer)}")
            return np.zeros((1,), dtype=np.float32)
        
    def compile_kernel(self, source: str, kernel_name: str) -> object:
        """CPU doesn't compile kernels."""
        raise NotImplementedError("CPU device does not support kernel compilation")
        
    def execute_kernel(self, compiled_kernel, grid_size: tuple, threadgroup_size: tuple, buffers: list):
        """CPU doesn't execute kernels."""
        raise NotImplementedError("CPU device does not support kernel execution")
        
    def synchronize(self):
        """CPU is always synchronized."""
        pass
        
    def get_capabilities(self) -> dict:
        """Return CPU capabilities."""
        return {
            "name": "CPU",
            "available": True
        }
        
    def is_buffer(self, data):
        """Everything on CPU is a numpy array or can be converted to one."""
        return isinstance(data, np.ndarray)

# Supported device types in order of preference
SUPPORTED_DEVICES = ["METAL", "OPENCL", "CPU"]

class DeviceManager:
    """
    Singleton class to manage GPU devices.
    Handles device discovery, selection, and caching.
    """
    def __init__(self):
        self._devices: Dict[str, Device] = {}
        self._available_devices: List[str] = []
        self._current_device: Optional[Device] = None
        self._initialized: bool = False
        self._opened_devices: Set[str] = set()
        self._device_info_printed: bool = False
        
        # Initialize the manager
        self._initialize()
        
        # Register cleanup
        atexit.register(self._cleanup)

    def _initialize(self) -> None:
        """Initialize all available devices and populate the device list."""
        if self._initialized:
            return
            
        # Try to initialize Metal on macOS
        if platform.system() == "Darwin" and HAS_METAL:
            try:
                metal_device = MetalDevice()
                if metal_device.is_available():
                    self._devices["METAL"] = metal_device
                    self._available_devices.append("METAL")
            except Exception as e:
                if os.getenv("DEBUG") == "1":
                    print(f"Metal initialization error: {e}")
                    traceback.print_exc()
        
        # Try to initialize OpenCL
        if CL_AVAILABLE:
            try:
                cl_device = OpenCLDevice()
                if cl_device.is_available():
                    self._devices["OPENCL"] = cl_device
                    self._available_devices.append("OPENCL")
            except Exception as e:
                if os.getenv("DEBUG") == "1":
                    print(f"OpenCL initialization error: {e}")
                    traceback.print_exc()
        
        # Always have CPU as fallback
        self._devices["CPU"] = CPUDevice()
        self._available_devices.append("CPU")
        
        self._initialized = True
    
    @property
    def default_device(self) -> Device:
        """Get the default device based on environment or availability."""
        if self._current_device is None:
            # Check environment variables first
            for device_name in SUPPORTED_DEVICES:
                if os.getenv(device_name) == "1" and device_name in self._available_devices:
                    self._current_device = self._devices.get(device_name)
                    break
            
            # If still not set, use the first available
            if self._current_device is None and self._available_devices:
                device_name = self._available_devices[0]
                if device_name in self._devices:
                    self._current_device = self._devices[device_name]
                    
            # If we still don't have a device, use CPU
            if self._current_device is None:
                self._current_device = self._devices.get("CPU")
                    
            # If we found a device, print info
            if self._current_device is not None and not self._device_info_printed:
                try:
                    device_info = self._current_device.get_capabilities()
                    # print(f"Using {device_info.get('name', 'unknown')}")
                    self._device_info_printed = True
                except Exception:
                    pass
                
        return self._current_device
    
    @functools.lru_cache(maxsize=16)
    def get_device(self, device_name: Optional[str] = None) -> Device:
        """Get a device by name or the default device if none specified."""
        if device_name is None:
            return self.default_device
            
        device_name = device_name.upper()
        if device_name in self._devices:
            device = self._devices[device_name]
            self._opened_devices.add(device_name)
            return device
            
        # Fall back to CPU if requested device not available
        print(f"Device {device_name} not available, falling back to CPU")
        return self._devices.get("CPU")
        
    def set_device(self, device: Union[Device, str]) -> None:
        """Set the current device."""
        if isinstance(device, str):
            device_name = device.upper()
            if device_name in self._devices:
                self._current_device = self._devices[device_name]
                self._device_info_printed = False
            else:
                print(f"Device {device_name} not available, falling back to CPU")
                self._current_device = self._devices.get("CPU")
                self._device_info_printed = False
        else:
            self._current_device = device
            self._device_info_printed = False
            
    def synchronize(self) -> None:
        """Synchronize the current device."""
        if self._current_device:
            self._current_device.synchronize()
    
    def get_available_devices(self) -> List[str]:
        """Get a list of available device names."""
        return self._available_devices.copy()
    
    def _cleanup(self) -> None:
        """Clean up resources for all opened devices."""
        for device_name in self._opened_devices:
            try:
                if device_name in self._devices:
                    # Free any resources
                    pass
            except Exception as e:
                if os.getenv("DEBUG") == "1":
                    print(f"Error cleaning up device {device_name}: {e}")

    # Buffer management helper methods
    def allocate_buffer(self, shape: Tuple[int, ...], dtype=np.float32) -> Any:
        """Allocate a buffer on the current device."""
        device = self.default_device
        if device is None:
            return np.zeros(shape, dtype=dtype)
            
        size = int(np.prod(shape))
        buffer = device.allocate_memory(size * np.dtype(dtype).itemsize)
        
        # Store metadata if the device supports it
        if hasattr(device, 'buffer_metadata'):
            buffer_id = id(buffer)
            device.buffer_metadata[buffer_id] = {
                'shape': shape,
                'dtype': dtype,
                'size': size
            }
            
        return buffer
    
    def upload_tensor(self, data: np.ndarray) -> Any:
        """Upload a tensor to the current device."""
        device = self.default_device
        if device is None:
            return data
        return device.upload_tensor(data)
    
    def download_tensor(self, buffer: Any) -> np.ndarray:
        """Download a tensor from the current device."""
        device = self.default_device
        if device is None:
            # If no device, return as is or extract data
            if hasattr(buffer, "data"):
                return buffer.data
            return buffer
            
        # If a Tensor was passed, extract its data attribute
        if hasattr(buffer, "data"):
            buffer = buffer.data
            
        return device.download_tensor(buffer)
    
    def is_buffer(self, data: Any) -> bool:
        """Check if data is a device tensor/buffer."""
        device = self.default_device
        if device is None: return False
        if hasattr(device, 'is_buffer'): return device.is_buffer(data)
        if hasattr(data, "length") and callable(data.length): return True
        if hasattr(data, "__pyobjc_object__") or str(type(data)).find('Buffer') >= 0: return True
        return False

# Create the singleton instance
_DEVICE_MANAGER = DeviceManager()

# Export functions for backward compatibility and easy access
get_device = _DEVICE_MANAGER.get_device
set_device = _DEVICE_MANAGER.set_device
upload_tensor = _DEVICE_MANAGER.upload_tensor
download_tensor = _DEVICE_MANAGER.download_tensor
is_buffer = _DEVICE_MANAGER.is_buffer
allocate_buffer = _DEVICE_MANAGER.allocate_buffer
synchronize = _DEVICE_MANAGER.synchronize
get_available_devices = _DEVICE_MANAGER.get_available_devices 