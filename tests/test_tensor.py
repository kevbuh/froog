import numpy as np
import torch
import unittest
from froog.tensor import Tensor
from froog.gradcheck import numerical_jacobian, gradcheck, jacobian

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class TestTensor(unittest.TestCase):

  def test_jacobian(self):
    W = np.random.RandomState(1337).random((10, 5))
    x = np.random.RandomState(7331).random((1, 10)) - 0.5

    torch_x = torch.tensor(x, requires_grad=True)
    torch_W = torch.tensor(W, requires_grad=True)
    torch_func = lambda x: torch.nn.functional.log_softmax(x.matmul(torch_W).relu(), dim=1)

    froog_x = Tensor(x)
    froog_W = Tensor(W)
    froog_func = lambda x: x.dot(froog_W).relu().logsoftmax()

    J = jacobian(froog_func, froog_x)
    PJ = torch.autograd.functional.jacobian(torch_func, torch_x).squeeze().numpy()
    NJ = numerical_jacobian(froog_func, froog_x)

    np.testing.assert_allclose(PJ, J, atol = 1e-5)
    np.testing.assert_allclose(PJ, NJ, atol = 1e-5)

    def test_froog():
      x = Tensor(x_init)
      W = Tensor(W_init)
      m = Tensor(m_init)
      out = x.dot(W).relu()
      out = out.logsoftmax()
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.data, x.grad, W.grad

    def test_pytorch():
      x = torch.tensor(x_init, requires_grad=True)
      W = torch.tensor(W_init, requires_grad=True)
      m = torch.tensor(m_init)
      out = x.matmul(W).relu()
      out = torch.nn.functional.log_softmax(out, dim=1)
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.detach().numpy(), x.grad, W.grad

    for x,y in zip(test_froog(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)

  def test_gradcheck(self):
    class TorchModel(torch.nn.Module):
      def __init__(self, weights_init):
        super(TorchModel, self).__init__()
        self.l1 = torch.nn.Linear(*weights_init.shape, bias = False)
        self.l1.weight = torch.nn.Parameter(torch.tensor(weights_init.T, requires_grad = True))

      def forward(self, x):
        return torch.nn.functional.log_softmax(self.l1(x).relu(), dim=1)

    x = np.random.RandomState(7331).random((1, 10)) - 0.5
    W = np.random.RandomState(1337).random((10, 5))

    tiny_x = Tensor(x)
    tiny_W = Tensor(W)
    froog_func = lambda x: x.dot(tiny_W).relu().logsoftmax()

    self.assertTrue(gradcheck(froog_func, tiny_x))
    self.assertTrue(gradcheck(froog_func, tiny_x)) 
    self.assertFalse(gradcheck(froog_func, tiny_x, eps = 0.1)) 

if __name__ == '__main__':
  unittest.main()