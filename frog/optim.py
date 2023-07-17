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
    def __init__(self, params, alpha=0.001, b1=0.9, b2=0.999, eps=10e-8):
        super(Adam, self).__init__(params)
        self.params = params

        self.alpha = alpha
        self.b1 = b1
        self.b2 = b2
        self.eps = eps # should be 1e-8?
        self.t = 0

        self.m = [np.zeros_like(t.data) for t in self.params ]
        self.v = [np.zeros_like(t.data) for t in self.params ]

    def step(self):
        for i, t in enumerate(self.params):
            self.t += 1

            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * t.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b1) * np.square(t.grad)
            
            m_hat = self.m[i] / (1. - self.b1 ** self.t)
            v_hat = self.v[i] / (1. - self.b2 ** self.t)
            
            t.data -=  self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)


      

