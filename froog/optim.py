import numpy as np

class Optimizer:
   def __init__(self, params):
      self.params = params

class SGD(Optimizer):
  def __init__(self, params, lr=0.001):
    super(SGD, self).__init__(params)
    self.lr = lr

  def step(self):
    for t in self.params:
      t.data -= self.lr * t.grad

class Adam(Optimizer):  
  """
  Default ADAM opimizer from https://arxiv.org/pdf/1412.6980.pdf algorithm
  """
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=10e-8):
    super(Adam, self).__init__(params)
    self.lr = lr
    self.b1 = b1
    self.b2 = b2
    self.eps = eps # should be 1e-8?
    self.t = 0

    self.m = [np.zeros_like(t.data) for t in self.params]
    self.v = [np.zeros_like(t.data) for t in self.params]

  def step(self):
    self.t += 1
    a = self.lr * (
      np.sqrt(1 - np.power(self.b2, self.t)) /
      (1 - np.power(self.b1, self.t)))
    for i,t in enumerate(self.params):
      self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * t.grad
      self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(t.grad)
      t.data -= a * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

class RMSprop(Optimizer):
  """
  This version has epsilon
  https://optimization.cbe.cornell.edu/index.php?title=RMSProp
  RMSprop divides the learning rate by an exponentially decaying average of squared gradients.

  Notes: 
  The reason RPROP doesn’t work is that it violates the central idea behind stochastic gradient descent, 
  which is when we have small enough learning rate, it averages the gradients over successive mini-batches.
  """
  def __init__(self, params, decay=0.9, lr=0.001, eps=1e-8):
    super(RMSprop, self).__init__(params)
    self.lr = lr
    self.decay = decay
    self.eps = eps
    self.v = [np.zeros_like(t.data) for t in self.params]

  def step(self):
    for i,t in enumerate(self.params):
      self.v[i] = self.decay * self.v[i] + (1-self.decay) * np.square(t.grad)
      t.data -= self.lr / (np.sqrt(self.v[i]) + self.eps) * t.grad