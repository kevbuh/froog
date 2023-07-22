import time
import cProfile
import unittest
from frog.tensor import Tensor
import pstats
import numpy as np

def profile_conv(bs, chans, conv, num_times=100):
  img = Tensor.zeros(bs, 1, 28, 28)
  conv = Tensor.randn(chans, 1, conv, conv)
  forward_pass_time, backward_pass_time = 0.0, 0.0
  for _ in range(num_times):
    conv1_start = time.time()
    out = img.conv2d(conv)
    conv1_end = time.time()
    g = out.mean().backward() # use mean to calculate gradients
    gradient_calc_end = time.time()

    forward_pass_time += (conv1_end-conv1_start)
    backward_pass_time += (gradient_calc_end-conv1_end)
  return forward_pass_time/num_times, backward_pass_time/num_times


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
  ps.print_stats(0.3)


class TestConvSpeed(unittest.TestCase):
  def test_3x3_conv(self):
    # warmup
    profile_conv(128, 16, 3, num_times=1)
    pr = start_profile()
    fwd_pass_avg, backward_pass_avg = profile_conv(128, 16, 3)
    stop_profile(pr)
    print(f"avg forward pass :  {float(fwd_pass_avg*1000):.2f} ms")
    print(f"avg backward pass: {float(backward_pass_avg*1000):.2f} ms")

  def test_mnist(self):
      # https://keras.io/examples/vision/mnist_convnet/
      conv = 3
      inter_chan, out_chan = 32, 64
      c1 = Tensor.randn(inter_chan,1,conv,conv)
      c2 = Tensor.randn(out_chan,inter_chan,conv,conv)
      l1 = Tensor.randn(out_chan*5*5, 10)

      cnt = 5
      fpt, bpt = 0.0, 0.0
      for i in range(1+cnt):
        et0 = time.time()
        x = Tensor.randn(128, 1, 28, 28)
        x = x.conv2d(c1).relu().maxpool2x2()
        x = x.conv2d(c2).relu().maxpool2x2()
        x = x.reshape(Tensor(np.array((x.shape[0], -1))))
        out = x.dot(l1).logsoftmax().mean()
        et1 = time.time()
        out.backward()
        et2 = time.time()
        if i == 0:
          pr = start_profile()
        else:
          fpt += (et1-et0)
          bpt += (et2-et1)

      stop_profile(pr, sort='time')

      print(f"avg forward pass: {float(fpt*1000/cnt):.3f} ms", float(fpt*1000/cnt))
      print(f"avg backward pass: {float(bpt*1000/cnt):.3f} ms", float(bpt*1000/cnt))

      stop_profile(pr, sort='time')

if __name__ == '__main__':
  unittest.main()