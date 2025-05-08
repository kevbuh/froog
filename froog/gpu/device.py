from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Tuple, Optional, Union, TypeVar, List, Dict, Callable

class Device(ABC):
    """Base device abstraction for GPU operations.
    
    This abstract class defines the interface that all device implementations
    must provide. It handles core operations like memory transfer, tensor creation,
    and computation required for neural network operations.
    """
    
    @abstractmethod
    def __init__(self):
        """Initialize the device."""
        pass
    
    # --- Core device properties and validation ---
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the device name."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the device is available for computation."""
        pass
    
    # --- Memory transfer operations ---
    
    @abstractmethod
    def tensor_to_device(self, data: np.ndarray) -> Any:
        """Convert CPU numpy array to device tensor."""
        pass
        
    @abstractmethod
    def tensor_to_cpu(self, tensor: Any) -> np.ndarray:
        """Convert a device tensor back to CPU numpy array."""
        pass
    
    @abstractmethod
    def is_device_tensor(self, data: Any) -> bool:
        """Check if data is a tensor on this device."""
        pass
    
    # --- Memory allocation operations ---
    
    @abstractmethod
    def buffer_new(self, shape: Tuple[int, ...]) -> Any:
        """Create a new uninitialized buffer with the given shape."""
        pass
    
    # --- Computation operations ---
    
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
    
    # --- Utility methods that can be implemented using core methods ---
    
    def get_size(self, x: Any) -> int:
        """Return the total number of elements in x.
        
        Default implementation works with shape tuples and arrays with shape attribute.
        Subclasses can override for better performance or device-specific behavior.
        """
        if hasattr(x, 'shape'):
            return int(np.prod(x.shape))
        elif isinstance(x, tuple):
            return int(np.prod(x))
        else:
            raise TypeError(f"Cannot determine size of {type(x)}")
    
    def buffer_zeros(self, shape: Tuple[int, ...]) -> Any:
        """Create a new buffer filled with zeros.
        
        Default implementation creates a buffer and fills it with zeros.
        Subclasses can override for better performance.
        """
        # Create numpy zeros and transfer to device
        return self.tensor_to_device(np.zeros(shape, dtype=np.float32))
    
    def buffer_like(self, x: Any) -> Any:
        """Create a new buffer with the same shape as x.
        
        Default implementation uses buffer_new with x's shape.
        """
        return self.buffer_new(x.shape)
    
    # --- Optional specialized operations ---
    
    def pooling_op(self, input: Any, kernel_size: Tuple[int, int], 
                  iter_op: str, result_op: str, init_val: Union[int, str] = 0) -> Any:
        """Apply a pooling operation to a tensor.
        
        This is a specialized operation that could be implemented
        using more primitive operations by the backend if needed.
        
        Args:
            input: Input tensor
            kernel_size: Size of pooling kernel (height, width)
            iter_op: Operation to perform for each element in the pooling window
            result_op: Operation to compute final result from accumulated values
            init_val: Initial value for the accumulator
            
        Returns:
            Output tensor after pooling
        """
        raise NotImplementedError(
            "This device does not implement pooling_op. "
            "Consider using a higher-level implementation."
        ) 