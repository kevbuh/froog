import numpy as np
import torch
import unittest
from froog.tensor import Tensor
from froog.optim import Adam, SGD, RMSprop

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

def step_froog(optim, kwargs={}):
  model = FrogNet()
  optim = optim([model.x, model.W], **kwargs)
  out = model.forward()
  out.backward()
  optim.step()
  return model.x.data, model.W.data

def step_pytorch(optim, kwargs={}):
  model = TorchNet()
  optim = optim([model.x, model.W], **kwargs)
  out = model.forward()
  out.backward()
  optim.step()
  return model.x.detach().numpy(), model.W.detach().numpy()

class FrogNet():
  def __init__(self):
    self.x = Tensor(x_init.copy())
    self.W = Tensor(W_init.copy())
    self.m = Tensor(m_init.copy())

  def forward(self):
    out = self.x.dot(self.W).relu()
    out = out.logsoftmax()
    out = out.mul(self.m).add(self.m).sum()
    return out

class TorchNet():
  def __init__(self):
    self.x = torch.tensor(x_init.copy(), requires_grad=True)
    self.W = torch.tensor(W_init.copy(), requires_grad=True)
    self.m = torch.tensor(m_init.copy())

  def forward(self):
    out = self.x.matmul(self.W).relu()
    out = torch.nn.functional.log_softmax(out, dim=1)
    out = out.mul(self.m).add(self.m).sum()
    return out

class TestOptim(unittest.TestCase):
  def test_adam(self):
    for x,y in zip(step_froog(Adam),step_pytorch(torch.optim.Adam)):
      np.testing.assert_allclose(x, y, atol=1e-6)

  def test_sgd(self):
    for x,y in zip(step_froog(SGD, kwargs={'lr': 0.001}), step_pytorch(torch.optim.SGD, kwargs={'lr': 0.001})):
      np.testing.assert_allclose(x, y, atol=1e-6)

  def test_rmsprop(self):
    for x,y in zip(step_froog(RMSprop, kwargs={'lr': 0.001, 'decay': 0.99}), step_pytorch(torch.optim.RMSprop, kwargs={'lr': 0.001, 'alpha': 0.99})):
      np.testing.assert_allclose(x, y, atol=1e-6)

if __name__ == '__main__':
  unittest.main()