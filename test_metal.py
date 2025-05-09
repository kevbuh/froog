#!/usr/bin/env python3
"""
Test script for Metal GPU operations in Froog.
This script creates some tensors, performs operations on them,
and verifies that Metal operations work correctly.
"""

import os
import sys
import platform
import numpy as np

def check_dependencies():
    """Check for required dependencies for Metal support"""
    print("Checking Metal dependencies...")
    
    # Verify we're on macOS
    if platform.system() != "Darwin":
        print("Error: Metal is only supported on macOS")
        print(f"Current system: {platform.system()}")
        return False
    
    # Check for PyObjC
    try:
        import objc
        print("PyObjC is installed")
    except ImportError:
        print("Error: PyObjC is not installed")
        print("Please install it with: pip install pyobjc")
        return False
        
    # Check for Metal
    try:
        import Metal
        print("Metal module is available")
    except ImportError:
        print("Error: Metal module is not available")
        print("This should be included with PyObjC on macOS")
        return False
    
    print("All Metal dependencies are satisfied")
    return True

from froog.tensor import Tensor
from froog.gpu import get_device

def main():
    print("\n--- Testing Metal GPU operations in Froog ---\n")
    
    # Check dependencies first
    has_dependencies = check_dependencies()
    if not has_dependencies:
        print("Warning: Metal dependencies are missing, test will run in CPU mode")
    
    # Get the device (this will initialize Metal if available)
    device = get_device()
    if device is None:
        print("No GPU device available. Test will run in CPU mode.")
    
    # Create tensors
    print("\nCreating tensors...")
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
    
    # Move to GPU if available
    if device is not None:
        print("Moving tensors to GPU...")
        a = a.to_gpu()
        b = b.to_gpu()
        print(f"Tensor a on GPU: {a.is_gpu}")
        print(f"Tensor b on GPU: {b.is_gpu}")
    
    # Test operations
    print("\nTesting basic operations:")
    print(f"a = {a.to_cpu().data}")
    print(f"b = {b.to_cpu().data}")
    
    # Addition
    c = a.add(b)
    print(f"a + b = {c.to_cpu().data}")
    
    # Multiplication
    d = a.mul(b)
    print(f"a * b = {d.to_cpu().data}")
    
    # Matrix multiplication
    e = a.dot(b)
    print(f"a @ b = {e.to_cpu().data}")
    
    # ReLU
    f = a.relu()
    print(f"relu(a) = {f.to_cpu().data}")
    
    # Test more complex operations if GPU is available
    if device is not None:
        print("\nTesting more operations on GPU:")
        
        # Create 4D tensor for pooling operations
        x = Tensor(np.random.rand(1, 3, 4, 4))
        x = x.to_gpu()
        print(f"Created 4D tensor with shape {x.shape}")
        
        # Max pooling
        pooled = x.max_pool2d()
        print(f"After max_pool2d: shape = {pooled.shape}")
        
        # More operations can be added here...
    
    print("\n--- Test completed ---")

if __name__ == "__main__":
    main() 