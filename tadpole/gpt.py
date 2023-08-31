# implement a gpt in under 200 lines
# use karpathy's libaries as a starting point

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
  """ 
  https://arxiv.org/abs/1607.06450
  scales and shifts the inputs within a layer so that they have a mean of 0 and a standard deviation of 1.
  """
  def __init__(self, ndim, bias):
    super().__init__()

    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None


  def forward(self, input):
    return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)