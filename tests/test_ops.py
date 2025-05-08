import numpy as np
from froog.tensor import Tensor, GPU
import torch
import unittest
import timeit
import functools

def helper_test_op(shape, torch_func, froog_func, atol=1e-7, grad_atol=1e-7, gpu=False, forward_only=False):
  torch_tensors = [torch.rand(x, requires_grad=True) for x in shape]
  froog_tensors = [Tensor(x.detach().numpy()) for x in torch_tensors]

  if gpu: froog_tensors = [x.to_gpu() for x in froog_tensors]

  out = torch_func(*torch_tensors)
  ret = froog_func(*froog_tensors)
  
  np.testing.assert_allclose(ret.to_cpu().data, out.detach().numpy(), atol=atol)

  if not forward_only:
    out.mean().backward()
    ret.mean().backward()

    for t, tt in zip(torch_tensors, froog_tensors):
      np.testing.assert_allclose(t.grad, tt.grad.to_cpu().data, atol=grad_atol)

  # test for speed
  # forward passes
  torch_fwd = timeit.Timer(functools.partial(torch_func, *torch_tensors)).timeit(5) * 1000/5
  froog_fwd = timeit.Timer(functools.partial(froog_func, *froog_tensors)).timeit(5) * 1000/5

  # backward passes
  if not forward_only:
    torch_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), torch_func, torch_tensors)).timeit(5) * 1000/5
    froog_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), froog_func, froog_tensors)).timeit(5) * 1000/5
  else:
    torch_fbp, froog_fbp = np.nan, np.nan

  print(f"shape: {repr(shape) : >32} torch/froog fwd: {torch_fwd:.2f}/{froog_fwd:.2f} ms ({float(froog_fwd/torch_fwd):.2f}x slower) bp: {torch_fbp - torch_fwd:.2f}/{froog_fbp - froog_fwd:.2f} ms ({float((froog_fbp - froog_fwd)/(torch_fbp - torch_fwd)):.2f}x slower)")

class TestOps(unittest.TestCase):
  gpu = False
  # **************** Add ****************
  def test_add(self): helper_test_op([(45,65), (45,65)], lambda x,y: x+y, Tensor.add, gpu=self.gpu)
  def test_sum(self): helper_test_op([(45,3)], lambda x: x.sum(), Tensor.sum, atol=1e-4, gpu=self.gpu)
  def test_broadcast_add(self): helper_test_op([(1,32,32,32), (1,32,1,1)], lambda x,y: x+y, Tensor.add, gpu=self.gpu, forward_only=True)
  # **************** Sub ****************
  def test_sub(self): helper_test_op([(45,65), (45,65)], lambda x,y: x-y, Tensor.sub, gpu=self.gpu)
  # **************** Mul ****************
  def test_mul(self): helper_test_op([(45,65), (45,65)], lambda x,y: x*y, Tensor.mul, gpu=self.gpu)
  # **************** Dot ****************
  def test_dot(self): helper_test_op([(45,65), (65,100)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-5, gpu=self.gpu)
  # **************** Div ****************
  def test_div(self): helper_test_op([(45,65), (45,65)], lambda x,y: x/y, Tensor.div, atol=1e-3, grad_atol=1e-3, gpu=self.gpu)
  # **************** Pow ****************
  def test_pow(self): helper_test_op([(45,65), (45,65)], lambda x,y: x**y, Tensor.pow, gpu=self.gpu)
  # **************** Sqrt ****************
  def test_sqrt(self): helper_test_op([(45,65)], lambda x: x.sqrt(), Tensor.sqrt, gpu=self.gpu)
  # **************** Conv ****************
  def test_conv2d(self):
    for bs in [1,8]:
      for cin in [1,2,3]:
        for groups in [1,3] if cin == 3 else [1]:
          for H in [1,2,5]:
            for W in [1,2,3,5]:
              helper_test_op([(bs,cin,11,28), (6,cin//groups,H,W)],
                lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
                lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=2e-5, grad_atol=2e-6, forward_only=self.gpu)
  def test_strided_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)], lambda x,w: torch.nn.functional.conv2d(x,w,stride=2).relu(), lambda x,w: Tensor.conv2d(x,w,stride=2).relu(), atol=2e-5, grad_atol=2e-6, forward_only=self.gpu)
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)], lambda x,w: torch.nn.functional.conv2d(x,w,stride=(2,1)).relu(), lambda x,w: Tensor.conv2d(x,w,stride=(2,1)).relu(), atol=2e-5, grad_atol=2e-6, forward_only=self.gpu)

  # **************** Max Pool ****************
  def test_maxpool_sizes(self):
    for size in [(2,2), (3,3), (3,2), (5,5), (5,1)]:
      helper_test_op([(32,2,110,28)],
        lambda x: torch.nn.functional.max_pool2d(x, kernel_size=size),
        lambda x: Tensor.max_pool2d(x, kernel_size=size), gpu=self.gpu, forward_only=self.gpu)
  def test_maxpool2x2(self): helper_test_op([(32,2,110,28)], lambda x: torch.nn.functional.max_pool2d(x, (2,2)), Tensor.max_pool2d, gpu=self.gpu, forward_only=self.gpu, atol=2e-5, grad_atol=2e-6)
  # **************** Avg Pool ****************
  def test_avgpool2x2(self): helper_test_op([(32,2,111,28)], lambda x: torch.nn.functional.avg_pool2d(x, (2,2)), Tensor.avg_pool2d, gpu=self.gpu, forward_only=self.gpu)
  # **************** Activations ****************
  def test_relu(self): helper_test_op([(45,65)], lambda x: x.relu(), Tensor.relu, gpu=self.gpu)
  # **************** Padding ****************
  def test_pad2d(self): helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (1,1,1,1)), lambda x: x.pad2d(padding=(1,1,1,1)), gpu=self.gpu, forward_only=True)

  # **************** Dropout ****************
  def test_dropout_eval_mode(self):
    helper_test_op([(10, 20)], lambda t: torch.nn.functional.dropout(t, p=0.5, training=False), lambda t: t.dropout(p=0.5, training=False), gpu=self.gpu)
    helper_test_op([(50, 50)], lambda t: torch.nn.functional.dropout(t, p=0.2, training=False), lambda t: t.dropout(p=0.2, training=False), gpu=self.gpu)
    helper_test_op([(10, 20)], lambda t: torch.nn.functional.dropout(t, p=0.0, training=False), lambda t: t.dropout(p=0.0, training=False), gpu=self.gpu)

  def test_dropout_train_mode_cpu(self):
    if self.gpu:
      self.skipTest("Training mode dropout test is CPU-only for now due to GPU random number generation.")
      return
    P_DROPOUT = 0.25
    SHAPE = (100, 200)

    # --- Test 1: Forward Pass Verification ---
    np.random.seed(1337)
    np_x = np.random.rand(*SHAPE).astype(np.float32)
    froog_x = Tensor(np_x.copy())
    np.random.seed(1337)
    froog_y = froog_x.dropout(p=P_DROPOUT, training=True)
    np.random.seed(1337)
    expected_unscaled_mask_np = (np.random.rand(*SHAPE) >= P_DROPOUT).astype(np.float32)
    # handle p=1.0 case for denominator as in the op
    denominator = (1.0 - P_DROPOUT) if P_DROPOUT < 1.0 else 1e-9
    expected_scaled_mask_np = expected_unscaled_mask_np / denominator
    expected_y_np = np_x * expected_scaled_mask_np
    np.testing.assert_allclose(froog_y.data, expected_y_np, atol=1e-7, err_msg="Dropout forward pass output does not match expected (CPU).")
    internal_scaled_mask_data = froog_y._ctx.saved_tensors[0]
    np.testing.assert_allclose(internal_scaled_mask_data, expected_scaled_mask_np, atol=1e-7, err_msg="Internal scaled mask in context does not match expected (CPU).")
    
    # --- Test 2: Backward Pass Verification ---
    np.random.seed(42)
    np_x_bp = np.random.rand(*SHAPE).astype(np.float32)
    froog_x_bp = Tensor(np_x_bp.copy())
    np.random.seed(42)
    froog_y_bp = froog_x_bp.dropout(p=P_DROPOUT, training=True)
    scaled_mask_for_bp = froog_y_bp._ctx.saved_tensors[0]
    froog_y_bp.mean().backward() 
    expected_grad_np = (1.0 / np.prod(SHAPE)) * scaled_mask_for_bp
    np.testing.assert_allclose(froog_x_bp.grad.data, expected_grad_np, atol=1e-7, err_msg="Dropout backward pass gradient does not match expected (CPU).")

if GPU:
  class TestOpsGPU(TestOps):
    gpu = True

if __name__ == '__main__':
  unittest.main(verbosity=2) 