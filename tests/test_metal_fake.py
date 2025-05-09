"""
Simple tests for Metal implementation with fake GPU mode.
"""
import os
import unittest
import numpy as np
from froog.tensor import Tensor
from froog import get_device

# Always enable fake GPU mode for these tests
os.environ["ALLOW_FAKE_GPU"] = "1"

class TestMetalFake(unittest.TestCase):
    """Test basic operations with fake Metal GPU support."""
    
    def test_add(self):
        """Test basic addition with fake GPU."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        
        # Move tensors to GPU
        a_gpu = a.to_gpu()
        b_gpu = b.to_gpu()
        
        # Verify GPU flag is set
        self.assertTrue(a_gpu.gpu)
        self.assertTrue(b_gpu.gpu)
        
        # Add on GPU
        c_gpu = a_gpu + b_gpu
        
        # Move result back to CPU
        c = c_gpu.to_cpu()
        
        # Verify result
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        np.testing.assert_allclose(c.data, expected)
    
    def test_mul(self):
        """Test multiplication with fake GPU."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        
        # Move tensors to GPU
        a_gpu = a.to_gpu()
        b_gpu = b.to_gpu()
        
        # Multiply on GPU
        c_gpu = a_gpu * b_gpu
        
        # Move result back to CPU
        c = c_gpu.to_cpu()
        
        # Verify result
        expected = np.array([4.0, 10.0, 18.0], dtype=np.float32)
        np.testing.assert_allclose(c.data, expected)
    
    def test_shape(self):
        """Test shape property with fake GPU."""
        a = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
        a_gpu = a.to_gpu()
        
        # Verify shapes match
        self.assertEqual(a.shape, a_gpu.shape)
    
    def test_tensor_creation(self):
        """Test creating tensors directly on GPU."""
        a = Tensor.randn(2, 3, 4).to_gpu()
        self.assertTrue(a.gpu)
        self.assertEqual(a.shape, (2, 3, 4))
        
        # Convert back to CPU and verify
        a_cpu = a.to_cpu()
        self.assertFalse(a_cpu.gpu)
        self.assertEqual(a_cpu.shape, (2, 3, 4))

if __name__ == "__main__":
    unittest.main() 