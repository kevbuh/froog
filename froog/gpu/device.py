from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Tuple, Optional, Union, TypeVar, List, Dict, Callable

class Device(ABC):
    """Abstract base class representing a generic GPU compute device."""
    
    @abstractmethod
    def allocate_memory(self, size: int):
        """Allocate `size` bytes on the GPU device memory and return a handle to the allocated buffer."""
        ...
    
    @abstractmethod
    def free_memory(self, buffer):
        """Free a previously allocated GPU buffer."""
        ...
    
    @abstractmethod
    def upload_tensor(self, host_array) -> object:
        """Copy data from a host (CPU) array to device memory. Returns a device buffer containing the data."""
        ...
    
    @abstractmethod
    def download_tensor(self, buffer) -> object:
        """Copy data from a device memory buffer back to a host array (e.g., NumPy array)."""
        ...
    
    @abstractmethod
    def compile_kernel(self, source: str, kernel_name: str) -> object:
        """Compile a compute kernel from source code. Returns an object representing the compiled kernel (e.g., a handle or pipeline state)."""
        ...
    
    @abstractmethod
    def execute_kernel(self, compiled_kernel, grid_size: tuple, threadgroup_size: tuple, buffers: list):
        """Launch a compiled kernel on the device. `grid_size` is the global work size (threads), `threadgroup_size` is the size of each thread group, and `buffers` is a list of device buffers to bind to the kernel."""
        ...
    
    @abstractmethod
    def synchronize(self):
        """Block until all pending device operations (kernels, memory transfers) have completed."""
        ...
    
    @abstractmethod
    def get_capabilities(self) -> dict:
        """Query device capabilities (e.g., name, total memory) and return them as a dictionary."""
        ...
