import numpy as np
from froog import Device, OpenCLDevice, get_device, set_device
from froog.tensor import Tensor

def test_device_abstraction():
    print("Testing device abstraction...")
    
    # Get the default device
    device = get_device()
    print(f"Default device: {device.name if device else 'None'}")
    
    # Test tensor creation and movement to GPU
    if device and device.is_available():
        try:
            # Create a tensor on CPU
            a = Tensor([1.0, 2.0, 3.0])
            print(f"Created tensor on CPU: {a.data}")
            
            # Move to GPU
            a_gpu = a.to_gpu()
            print(f"Moved tensor to GPU, GPU flag: {a_gpu.gpu}")
            
            # Move back to CPU
            a_cpu = a_gpu.to_cpu()
            print(f"Moved tensor back to CPU: {a_cpu.data}")
            
            # Test basic operations on GPU
            b = Tensor([4.0, 5.0, 6.0]).to_gpu()
            c = a_gpu + b
            print(f"Result of addition on GPU (after moving to CPU): {c.to_cpu().data}")
            
            print("All operations completed successfully!")
        except Exception as e:
            print(f"Error during GPU operations: {e}")
    else:
        print("No GPU device available for testing")

if __name__ == "__main__":
    test_device_abstraction()