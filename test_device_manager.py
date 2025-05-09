import numpy as np
import froog
from froog import (
    get_device, set_device, upload_tensor, download_tensor, 
    is_device_tensor, allocate_buffer, synchronize
)
from froog.tensor import Tensor

def test_device_manager():
    """Test the new device manager functionality."""
    print("Testing device manager...")
    
    # Get the default device
    device = get_device()
    print(f"Default device: {device.name if device else 'None'}")
    
    # Create a simple tensor
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = Tensor(a, gpu=True)
    
    # Verify it's on GPU
    print(f"Tensor is on GPU: {t.is_gpu}")
    
    # Convert it back to CPU
    t_cpu = t.to_cpu()
    print(f"CPU tensor data: {t_cpu.data}")
    
    # Test the direct buffer functions
    print("Testing buffer functions...")
    
    # Upload and download a buffer directly
    buffer = upload_tensor(a)
    print(f"Buffer is device tensor: {is_device_tensor(buffer)}")
    
    # Download the tensor
    data = download_tensor(buffer)
    print(f"Downloaded data: {data}")
    
    # Test synchronize
    synchronize()
    print("Device synchronized successfully")
    
    # Test allocate buffer
    allocated_buffer = allocate_buffer((2, 2))
    print(f"Allocated buffer is device tensor: {is_device_tensor(allocated_buffer)}")
    
    # Simple computation test
    b = Tensor([2.0, 3.0, 4.0], gpu=True)
    c = t + b
    print(f"Result on GPU: {c.is_gpu}")
    print(f"Result as CPU: {c.to_cpu().data}")
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_device_manager() 