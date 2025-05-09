from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Tuple, Optional, Union, TypeVar, List, Dict, Callable

class Device(ABC):
    """Base device abstraction for GPU operations."""
    
    @abstractmethod
    def __init__(self):
        """Initialize the device."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the device name."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the device is available."""
        pass
        
    @abstractmethod
    def tensor_to_device(self, data: np.ndarray) -> Any:
        """Convert CPU data to device."""
        pass
        
    @abstractmethod
    def tensor_to_cpu(self, tensor: Any) -> np.ndarray:
        """Convert a device tensor to CPU."""
        pass
    
    @abstractmethod
    def is_device_tensor(self, data: Any) -> bool:
        """Check if data is a device tensor."""
        pass
    
    @abstractmethod
    def get_size(self, x: Any) -> int:
        """Return the total number of elements in x."""
        pass
    
    @abstractmethod
    def buffer_new(self, shape: Tuple[int, ...]) -> Any:
        """Create a new empty buffer with the given shape."""
        pass
    
    @abstractmethod
    def buffer_zeros(self, shape: Tuple[int, ...]) -> Any:
        """Create a new buffer filled with zeros."""
        pass
    
    @abstractmethod
    def buffer_like(self, x: Any) -> Any:
        """Create a new buffer with the same shape as x."""
        pass
    
    @abstractmethod
    def build_program(self, code: str) -> Any:
        """Build a program/kernel from source code."""
        pass
    
    @abstractmethod
    def binary_op(self, code: str, x: Any, y: Any) -> Any:
        """Apply a binary operation to two tensors."""
        pass
    
    @abstractmethod
    def unary_op(self, code: str, x: Any) -> Any:
        """Apply a unary operation to a tensor."""
        pass
    
    @abstractmethod
    def pooling_op(self, input: Any, kernel_size: Tuple[int, int], iter_op: str, result_op: str, init_val: Union[int, str] = 0) -> Any:
        """Apply a pooling operation to a tensor."""
        pass