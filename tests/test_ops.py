import torch
import numpy as np
import unittest
from frog.tensor import Tensor
import timeit
import functools

def test_op(shape, torch_func, frog_func, atol=1e-7, grad_atol=1e-7):
  torch_tensors = [torch.rand(x, requires_grad=True) for x in shape]
  frog_tensors = [Tensor(x.detach().numpy()) for x in torch_tensors]

  out = torch_func(*torch_tensors)
  ret = frog_func(*frog_tensors)
  
  np.testing.assert_allclose(ret.data, out.detach().numpy(), atol=atol)

  out.mean().backward()
  ret.mean().backward()

  for t, tt in zip(torch_tensors, frog_tensors):
    np.testing.assert_allclose(t.grad, tt.grad, atol=grad_atol)

  # test for speed
  # forward passes
  torch_fwd = timeit.Timer(functools.partial(torch_func, *torch_tensors)).timeit(5) * 1000/5
  frog_fwd = timeit.Timer(functools.partial(frog_func, *frog_tensors)).timeit(5) * 1000/5

  # backward passes
  torch_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), torch_func, torch_tensors)).timeit(5) * 1000/5
  frog_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), frog_func, frog_tensors)).timeit(5) * 1000/5

  print(f"shape: {repr(shape) : >32} torch/frog fwd: {torch_fwd:.2f}/{frog_fwd:.2f} ms ({float(frog_fwd/torch_fwd):.2f}x slower) bp: {torch_fbp - torch_fwd:.2f}/{frog_fbp - frog_fwd:.2f} ms ({float((frog_fbp - frog_fwd)/(torch_fbp - torch_fwd)):.2f}x slower)")


class TestOps(unittest.TestCase):
  def test_conv2d(self):
    for bs in [1,128]:
      for cin in [1,2,3]:
        for H in [2,3,5]:
          for W in [2,3,5]:
            test_op([(bs,cin,10,7), (4,cin,H,W)], 
                    lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
                    lambda x,w: Tensor.conv2d(x,w).relu(), 
                    atol=2e-5, 
                    grad_atol=2e-6)

  def test_maxpool2x2(self):
    test_op([(32,2,110,28)], lambda x: torch.nn.functional.max_pool2d(x, (2,2)), Tensor.max_pool2d)

  def test_avgpool2x2(self):
    test_op([(32,2,111,28)], lambda x: torch.nn.functional.avg_pool2d(x, (2,2)), Tensor.avg_pool2d)

if __name__ == '__main__':
  unittest.main(verbosity=2) # TODO: what does verbosity do? 