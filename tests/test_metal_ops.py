"""
Tests for Metal operations with actual Metal GPU support.
"""
import os
import unittest
import numpy as np
from froog.tensor import Tensor
from froog import get_device

class TestMetalOps(unittest.TestCase):
    """Test operations with actual Metal GPU support."""
    
    def setUp(self):
        # Skip tests if Metal is not available
        device = get_device()
        if device is None or device.__class__.__name__ != "MetalDevice":
            self.skipTest("Metal device not available")
    
    def test_add(self):
        """Test addition with Metal GPU."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        
        # Move tensors to GPU
        a_gpu = a.to_gpu()
        b_gpu = b.to_gpu()
        
        # Add on GPU
        c_gpu = a_gpu + b_gpu
        
        # Move result back to CPU
        c = c_gpu.to_cpu()
        
        # Verify result
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        np.testing.assert_allclose(c.data, expected)
    
    def test_matmul(self):
        """Test matrix multiplication with Metal GPU."""
        # Create matrices
        a = Tensor(np.random.randn(3, 4).astype(np.float32))
        b = Tensor(np.random.randn(4, 5).astype(np.float32))
        
        # Expected result on CPU
        expected = a.data @ b.data
        
        # Move to GPU
        a_gpu = a.to_gpu()
        b_gpu = b.to_gpu()
        
        # Multiply on GPU
        c_gpu = a_gpu.dot(b_gpu)
        
        # Get CPU result
        c = c_gpu.to_cpu()
        
        # Verify result
        np.testing.assert_allclose(c.data, expected, rtol=1e-5, atol=1e-5)
    
    def test_relu(self):
        """Test ReLU with Metal GPU."""
        # Create tensor with positive and negative values
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        a = Tensor(data)
        
        # Expected result
        expected = np.maximum(data, 0)
        
        # Move to GPU
        a_gpu = a.to_gpu()
        
        # Apply ReLU on GPU
        b_gpu = a_gpu.relu()
        
        # Get CPU result
        b = b_gpu.to_cpu()
        
        # Verify result
        np.testing.assert_allclose(b.data, expected)
    
    def test_basic_nn(self):
        """Test a basic neural network with Metal GPU."""
        # Skip test in fake GPU mode
        if os.getenv("ALLOW_FAKE_GPU") == "1":
            self.skipTest("Skipping neural network test in fake GPU mode")
        
        # Create a simple 2-layer network
        x = Tensor(np.random.randn(10, 5).astype(np.float32))
        w1 = Tensor(np.random.randn(5, 8).astype(np.float32))
        w2 = Tensor(np.random.randn(8, 1).astype(np.float32))
        
        # Forward pass on CPU
        cpu_h1 = x.dot(w1).relu()
        cpu_out = cpu_h1.dot(w2)
        
        # Move to GPU
        x_gpu = x.to_gpu()
        w1_gpu = w1.to_gpu()
        w2_gpu = w2.to_gpu()
        
        # Forward pass on GPU
        gpu_h1 = x_gpu.dot(w1_gpu).relu()
        gpu_out = gpu_h1.dot(w2_gpu)
        
        # Get CPU result
        gpu_result = gpu_out.to_cpu()
        
        # Verify results match
        np.testing.assert_allclose(gpu_result.data, cpu_out.data, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    unittest.main() 