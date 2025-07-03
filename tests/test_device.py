import numpy as np
import pytest
from froog import get_device
from froog.tensor import Tensor

def test_device_abstraction():
    device = get_device()
    # Skip the test if no GPU is present
    if device is None or device.name == "CPU":
        pytest.skip("No GPU device available for testing")

    assert device.is_available(), "GPU device reported as unavailable"

    # Create a tensor on CPU
    a = Tensor([1.0, 2.0, 3.0])
    assert np.array_equal(a.data, [1.0, 2.0, 3.0]), "Tensor data mismatch on CPU"

    # Move to GPU
    a_gpu = a.to_gpu()
    assert a_gpu.gpu, "Tensor not moved to GPU"

    # Move back to CPU
    a_cpu = a_gpu.to_cpu()
    assert np.array_equal(a_cpu.data, [1.0, 2.0, 3.0]), "Tensor data mismatch after moving back to CPU"

    # Test basic operations on GPU
    b = Tensor([4.0, 5.0, 6.0]).to_gpu()
    c = a_gpu + b
    assert np.array_equal(c.to_cpu().data, [5.0, 7.0, 9.0]), "Addition result mismatch on GPU"
