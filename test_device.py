#!/usr/bin/env python
# Test script for the device abstraction in froog

import os
import sys
import numpy as np

# Set debug mode to see initialization messages
os.environ["DEBUG"] = "1"

# Import froog modules after setting environment variables
from froog.tensor import Tensor
from froog.gpu import get_device

def run_tests():
    # Check if we have a device
    device = get_device()
    print(f"Using device: {device.__class__.__name__ if device else 'None'}")
    
    if device:
        print(f"Device capabilities: {device.get_capabilities()}")
    else:
        print("No GPU device available")
    
    # Test basic tensor operations on the device
    print("\n--- Testing basic tensor operations ---")
    
    # Create a tensor on CPU
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    print(f"Tensor a (CPU): {a.data}")
    
    # Move to GPU
    a_gpu = a.to_gpu()
    print(f"Tensor a (GPU): {a_gpu.data}")
    print(f"Is on GPU: {a_gpu.is_gpu}")
    
    # Move back to CPU
    a_cpu = a_gpu.to_cpu()
    print(f"Tensor a (back to CPU): {a_cpu.data}")
    
    # Test a simple operation
    print("\n--- Testing tensor addition ---")
    print(f"Available GPU operations: {list(Tensor.ops_gpu.keys())}")
    
    b = Tensor([5.0, 6.0, 7.0, 8.0], gpu=True)
    try:
        c = a_gpu + b
        print(f"Result of addition (on CPU): {c.to_cpu().data}")
    except Exception as e:
        print(f"Addition failed: {e}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    run_tests() 