#!/usr/bin/env python
# Script to run GPU tests with fake GPU mode

import os
import sys
import subprocess

def main():
    # Set environment variables for testing
    env = os.environ.copy()
    env["ALLOW_FAKE_GPU"] = "1"
    env["DEBUG"] = "1"
    
    # First, run our device test
    print("=== Running device test ===")
    subprocess.run([sys.executable, "test_device.py"], env=env)
    
    # Run the actual GPU tests
    print("\n=== Running GPU tests with fake GPU mode ===")
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_ops.py::TestOpsGPU", "-v"], env=env)
    
    print("\n=== Running model GPU tests ===")
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_models.py::TestMNIST::test_conv_gpu", 
                   "tests/test_models.py::TestMNIST::test_sgd_gpu", "-v"], env=env)

if __name__ == "__main__":
    main() 