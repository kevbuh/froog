import time
import cProfile
import unittest
from frog.tensor import Tensor
import pstats

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
    

class TestConvSpeed(unittest.TestCase):
  def test_3x3_conv(self):
    # multiplying by 1e9 converts seconds to nanoseconds
    # 1e-6 refers to a microsecond (μs)
    pr = cProfile.Profile(timer=lambda: int(time.time()*1e9), timeunit=1e-6) 

    pr.enable()
    fwd_pass_avg, backward_pass_avg = profile_conv(128, 16, 3)
    pr.disable()

    ps = pstats.Stats(pr)
    ps.strip_dirs()
    ps.sort_stats('cumtime')
    ps.print_stats(0.3)

    print(f"avg forward pass :  {float(fwd_pass_avg*1000):.2f} ms")
    print(f"avg backward pass: {float(backward_pass_avg*1000):.2f} ms")

if __name__ == '__main__':
  unittest.main()