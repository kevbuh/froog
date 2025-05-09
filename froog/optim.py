#  _______  ______    _______  _______  _______ 
# |       ||    _ |  |       ||       ||       |
# |    ___||   | ||  |   _   ||   _   ||    ___|
# |   |___ |   |_||_ |  | |  ||  | |  ||   | __ 
# |    ___||    __  ||  |_|  ||  |_|  ||   ||  |
# |   |    |   |  | ||       ||       ||   |_| |
# |___|    |___|  |_||_______||_______||_______|

import numpy as np
from typing import List
from froog.tensor import Tensor

class Optimizer:
  def __init__(self, params: List[Tensor]) -> None:
    self.params = params

class SGD(Optimizer):
  """
  Stochastic Gradient Descent
  """
  def __init__(self, params: List[Tensor], lr: float = 0.001, weight_decay: float = 0, clip_value: float = 0) -> None:
    super(SGD, self).__init__(params)
    self.lr = Tensor([lr], gpu=params[0].gpu if params else False)
    self.weight_decay = weight_decay
    self.clip_value = clip_value

  def step(self) -> None:
    for t in self.params:
      if t.grad is None:
        continue
      
      if t.gpu:
        from froog.gpu import get_device, download_tensor, upload_tensor
        
        # device = get_device()
        t_cpu = download_tensor(t)
        grad_cpu = download_tensor(t.grad)
        lr_cpu = download_tensor(self.lr)
        
        if self.weight_decay > 0:
          grad_cpu += self.weight_decay * t_cpu
        
        if self.clip_value > 0:
          grad_cpu = np.clip(grad_cpu, -self.clip_value, self.clip_value)
        
        t_cpu -= grad_cpu * lr_cpu
        t.data = upload_tensor(t_cpu)
      else:
        if self.weight_decay > 0:
          t.grad.data += self.weight_decay * t.data
        
        if self.clip_value > 0:
          t.grad.data = np.clip(t.grad.data, -self.clip_value, self.clip_value)
        
        t -= t.grad * self.lr

class Adam(Optimizer):  
  """
  Default ADAM optimizer from https://arxiv.org/pdf/1412.6980.pdf algorithm
  """
  def __init__(self, params: List[Tensor], lr: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, max_grad: float = 10.0) -> None:
    super(Adam, self).__init__(params)
    self.lr = lr
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
    self.t = 0
    self.max_grad = max_grad
    self.on_gpu = any(t.gpu for t in self.params if t is not None)
    
    if self.on_gpu:
      from froog.gpu import download_tensor
      self.m = [np.zeros_like(download_tensor(t.data)) for t in self.params]
      self.v = [np.zeros_like(download_tensor(t.data)) for t in self.params]
    else:
      self.m = [np.zeros_like(t.data) for t in self.params]
      self.v = [np.zeros_like(t.data) for t in self.params]

  def step(self) -> None:
    from froog.gpu import download_tensor, upload_tensor
    
    self.t += 1
    a = self.lr * (np.sqrt(1 - np.power(self.b2, self.t)) / (1 - np.power(self.b1, self.t)))
      
    for i, t in enumerate(self.params):
      if t.grad is None:
        continue
        
      if t.gpu:
        try:
          t_data_cpu = download_tensor(t.data)
          grad_cpu = download_tensor(t.grad.data)
          
          if np.isnan(grad_cpu).any() or np.isinf(grad_cpu).any():
            print(f"Warning: NaN or Inf detected in gradients for parameter {i}")
            grad_cpu = np.nan_to_num(grad_cpu, nan=0.0, posinf=self.max_grad, neginf=-self.max_grad)
          
          if self.max_grad > 0:
            grad_cpu = np.clip(grad_cpu, -self.max_grad, self.max_grad)
          
          self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad_cpu
          self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(grad_cpu)
          
          denom = np.sqrt(self.v[i]) + self.eps
          update = a * self.m[i] / denom
          
          if np.isnan(update).any() or np.isinf(update).any():
            print(f"Warning: NaN or Inf detected in update for parameter {i}")
            max_update = np.finfo(np.float32).max / 100
            update = np.nan_to_num(update, nan=0.0, posinf=max_update, neginf=-max_update)
          
          t_data_cpu -= update
          
          if np.isnan(t_data_cpu).any() or np.isinf(t_data_cpu).any():
            print(f"Warning: NaN or Inf detected in parameter {i} after update")
            max_val = np.finfo(np.float32).max / 10
            t_data_cpu = np.nan_to_num(t_data_cpu, nan=0.0, posinf=max_val, neginf=-max_val)
          
          t.data = upload_tensor(t_data_cpu)
        except Exception as e:
          print(f"Error in Adam update for GPU tensor {i}: {e}")
          continue
      else:
        if self.max_grad > 0:
          np.clip(t.grad.data, -self.max_grad, self.max_grad, out=t.grad.data)
          
        self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * t.grad.data
        self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(t.grad.data)
        t.data -= a * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

class RMSprop(Optimizer):
  """
  RMSprop optimizer with epsilon for numerical stability.
  """
  def __init__(self, params: List[Tensor], decay: float = 0.9, lr: float = 0.001, eps: float = 1e-8) -> None:
    super(RMSprop, self).__init__(params)
    self.lr = lr
    self.decay = decay
    self.eps = eps
    self.v: List[np.ndarray] = [np.zeros_like(t.data) for t in self.params]

  def step(self) -> None:
    for i, t in enumerate(self.params):
      self.v[i] = self.decay * self.v[i] + (1 - self.decay) * np.square(t.grad.data)
      t.data -= self.lr / (np.sqrt(self.v[i]) + self.eps) * t.grad.data