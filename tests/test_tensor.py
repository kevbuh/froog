import numpy as np
import torch
import unittest
from frog.tensor import Tensor, Conv2D
from frog.gradcheck import numerical_jacobian, gradcheck

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class TestTensor(unittest.TestCase):
  def test_backward_pass(self):
    def test_tinygrad():
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

    for x,y in zip(test_tinygrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)

  def test_conv2d(self):
    x = torch.randn((5,2,10,7), requires_grad=True)
    w = torch.randn((4,2,3,3), requires_grad=True)
    xt = Tensor(x.detach().numpy())
    wt = Tensor(w.detach().numpy())

    out = torch.nn.functional.conv2d(x,w)
    ret = Conv2D.apply(Conv2D, xt, wt)
    np.testing.assert_allclose(ret.data, out.detach().numpy(), atol=1e-5)

    out.mean().backward()
    ret.mean().backward()

    np.testing.assert_allclose(w.grad, wt.grad, atol=1e-5)
    np.testing.assert_allclose(x.grad, xt.grad, atol=1e-5)

  def test_gradcheck(self):
    class FrogModel:
      def __init__(self, weights_init):
        self.l1 = Tensor(weights_init)
      def forward(self, x):
        return x.dot(self.l1).relu().logsoftmax()

    class TorchModel(torch.nn.Module):
      def __init__(self, weights_init):
        super(TorchModel, self).__init__()
        self.l1 = torch.nn.Linear(*weights_init.shape, bias = False)
        self.l1.weight = torch.nn.Parameter(torch.tensor(weights_init.T, requires_grad = True))

      def forward(self, x):
        return torch.nn.functional.log_softmax(self.l1(x).relu(), dim=1)

    layer_weights = np.random.RandomState(1337).random((10, 5))
    input_data = np.random.RandomState(7331).random((1, 10)) - 0.5 # TODO: why -0.5?

    torch_input = torch.tensor(input_data, requires_grad = True)
    torch_model = TorchModel(layer_weights)
    torch_out = torch_model(torch_input)
    J_sum = torch.autograd.grad(list(torch_out[0]), torch_input)[0].squeeze().numpy()

    frog_model = FrogModel(layer_weights)
    tiny_input = Tensor(input_data)
    tiny_out = frog_model.forward(tiny_input)

    NJ = numerical_jacobian(frog_model, tiny_input)
    NJ_sum = NJ.sum(axis = -1) # TODO: why this axis?

    # test frog grad
    np.testing.assert_allclose(J_sum, NJ_sum, atol = 1e-5)
    # test gradcheck
    gradcheck_test, _, _ = gradcheck(frog_model, tiny_input)
    self.assertTrue(gradcheck_test)


if __name__ == '__main__':
  unittest.main()