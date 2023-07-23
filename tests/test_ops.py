import numpy as np
import torch
import unittest
from frog.tensor import Tensor

class TestOps(unittest.TestCase):
  def test_conv2d(self):
    for cin in [1,2,3]:
      for H in [2,3,5]:
        for W in [2,3,5]:
          x = torch.randn((5,cin,10,7), requires_grad=True)
          w = torch.randn((4,cin,H,W), requires_grad=True)
          xt = Tensor(x.detach().numpy())
          wt = Tensor(w.detach().numpy())

          out = torch.nn.functional.conv2d(x,w)
          ret = Tensor.conv2d(xt, wt)
          np.testing.assert_allclose(ret.data, out.detach().numpy(), atol=1e-5)

          out.relu().mean().backward()
          ret.relu().mean().backward()

          np.testing.assert_allclose(w.grad, wt.grad, atol=1e-7)
          np.testing.assert_allclose(x.grad, xt.grad, atol=1e-7)

  def test_max_pool2d(self):
    x = torch.randn((5,2,10,8), requires_grad=True)
    x_frog = Tensor(x.detach().numpy())

    # in frog 
    ret = x_frog.max_pool2d()
    assert ret.shape == (5,2,10//2,8//2) # TODO: why this shape???
    ret.mean().backward()

    # in torch
    out = torch.nn.MaxPool2d((2,2))(x)
    out.mean().backward()

    # forward and backward the same
    np.testing.assert_allclose(ret.data, out.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(x.grad, x_frog.grad, atol=1e-5)

if __name__ == '__main__':
  unittest.main()