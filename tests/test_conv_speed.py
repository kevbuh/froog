import time
import cProfile
import unittest
from froog.tensor import Tensor
import pstats
import numpy as np
import torch

def start_profile():
  import time
  # multiplying by 1e9 converts seconds to nanoseconds
  # 1e-6 refers to a microsecond (μs)
  pr = cProfile.Profile(timer=lambda: int(time.time()*1e9), timeunit=1e-6)
  pr.enable()
  return pr


def stop_profile(pr, sort='cumtime'):
  pr.disable()
  ps = pstats.Stats(pr)
  ps.strip_dirs()
  ps.sort_stats(sort)
  ps.print_stats(0.2) # print only top 20% of time consuming fn calls


class TestConvSpeed(unittest.TestCase):
  def test_mnist(self):
    # https://keras.io/examples/vision/mnist_convnet/
    conv = 3
    inter_chan, out_chan = 32, 64

    # ****** torch baseline ******
    
    torch.backends.mkldnn.enabled = False                             # disables the use of MKL-DNN 
    conv = 3
    intern_chan, out_chan = 32, 64
    num_time = 5
    c1 = torch.rand(inter_chan,1,conv,conv, requires_grad=True)
    c2 = torch.randn(out_chan, inter_chan, conv, conv, requires_grad=True)
    l1 = torch.randn(out_chan*5*5,10, requires_grad=True)

    c2d = torch.nn.functional.conv2d
    mp = torch.nn.MaxPool2d((2,2))
    lsm = torch.nn.LogSoftmax(dim=1)

    with torch.autograd.profiler.profile(record_shapes=True) as tprof: # enables the collection of CPU and CUDA
      cnt = num_time
      fpt, bpt = 0.0, 0.0
      for i in range(cnt):
        et0 = time.time()
        x = torch.randn(128,1,28,28, requires_grad=True)
        x = mp(c2d(x,c1).relu())
        x = mp(c2d(x,c2).relu())
        x = x.reshape(x.shape[0], -1)
        out = lsm(x.matmul(l1))
        out = out.mean()
        et1 = time.time()
        out.backward() 
        et2=time.time()
        fpt += (et1-et0)
        bpt += (et2-et1)

    self.fpt_baseline = (fpt*1000)/cnt
    self.bpt_baseline = (bpt*1000)/cnt
    print(f"avg torch forward pass : {self.fpt_baseline:.3f} ms")
    print(f"avg torch backward pass: {self.bpt_baseline:.3f} ms")
    print(tprof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    # ****** froog results ******

    c1 = Tensor(c1.detach().numpy()) # detach from torch, turn into numpy array
    c2 = Tensor(c2.detach().numpy())
    l1 = Tensor(l1.detach().numpy())

    cnt = num_time
    fpt, bpt = 0.0, 0.0
    for i in range(1+cnt):
      et0 = time.time()
      x = Tensor.randn(128, 1, 28, 28)
      x = x.conv2d(c1).relu().avg_pool2d()
      x = x.conv2d(c2).relu().max_pool2d()
      x = x.reshape(shape=(x.shape[0], -1))
      out = x.dot(l1).logsoftmax()
      out = out.mean()
      et1 = time.time()
      out.backward()
      et2 = time.time()
      if i == 0:
        pr = start_profile()
      else:
        fpt += (et1-et0)
        bpt += (et2-et1)

    stop_profile(pr, sort='time')
    fpt = fpt*1000/cnt
    bpt = bpt*1000/cnt
    print(f"avg froog forward pass:  {float(fpt):.3f} ms, {float(fpt/self.fpt_baseline):.2f}x off baseline of {self.fpt_baseline:.3f} ms")
    print(f"avg froog backward pass: {float(bpt):.3f} ms, {float(fpt/self.bpt_baseline):.2f}x off baseline of {self.bpt_baseline:.3f} ms")
  
    
if __name__ == '__main__':
  unittest.main()