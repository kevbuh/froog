import numpy as np
from tqdm import trange
from froog.tensor import Tensor
from froog import get_device
from froog.utils import fetch_mnist
from froog.ops import Linear
import froog.optim as optim
import unittest
import os

# Check if GPU is available
HAS_GPU = get_device() is not None and get_device().name != "CPU"

# ********* load the mnist dataset *********
X_train, Y_train, X_test, Y_test = fetch_mnist()

# ********* creating a simple mlp *********
class SimpleMLP:
  def __init__(self):
    # 784 pixel inputs -> 128 -> 10 output
    # Initialize with better weight scaling for better convergence
    # Using Xavier/Glorot initialization: scale ~ sqrt(2 / (fan_in + fan_out))
    
    # First layer: fan_in = 784, fan_out = 128
    w1_scale = np.sqrt(2.0 / (784 + 128))
    self.l1 = Tensor(Linear(784, 128))
    self.l1.data = self.l1.data * w1_scale
    
    # Second layer: fan_in = 128, fan_out = 10
    w2_scale = np.sqrt(2.0 / (128 + 10))
    self.l2 = Tensor(Linear(128, 10))
    self.l2.data = self.l2.data * w2_scale

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()
  
  def parameters(self):
    return [self.l1, self.l2]

class SimpleConvNet:
  def __init__(self):
    # from https://keras.io/examples/vision/mnist_convnet/
    conv = 3
    inter_chan, out_chan = 8, 16 # for speed
    self.c1 = Tensor(Linear(inter_chan,1,conv,conv))                # (num_filters, color_channels, kernel_h, kernel_w)
    self.c2 = Tensor(Linear(out_chan,inter_chan,conv,conv))         # (28-conv+1)(28-conv+1) since kernel isn't padded
    self.l1 = Tensor(Linear(out_chan*5*5, 10))                      # MNIST output is 10 classes

  def forward(self, x):
    x = x.reshape(shape=(-1, 1, 28, 28))                            # get however many number of imgs in batch
    x = x.conv2d(self.c1).relu().max_pool2d()                       # pass through layer 1 first
    x = x.conv2d(self.c2).relu().max_pool2d()                       # pass through layer 2
    x = x.reshape(shape=[x.shape[0], -1])                           # then go down to mlp
    return x.dot(self.l1).logsoftmax()                              # softmax to get probs   

  def parameters(self):
    return [self.l1, self.c1, self.c2]

def train(model, optimizer, steps, BS=128, gpu=False):
  losses, accuracies = [], []

  for step in (t := trange(steps, disable=os.getenv('CI') is not None)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    x = Tensor(X_train[samp].reshape((-1, 28*28)).astype(np.float32), gpu=gpu)
    Y = Y_train[samp]
    y = np.zeros((len(samp),10), np.float32)
    y[range(y.shape[0]),Y] = -10.0
    y = Tensor(y, gpu=gpu)

    model_outputs = model.forward(x)
    loss = model_outputs.mul(y).mean()
    loss.backward()
    optimizer.step()

    pred = np.argmax(model_outputs.to_cpu().data, axis=1)
    accuracy = (pred == Y).mean()

    loss_value = loss.to_cpu().data
    losses.append(loss_value)
    accuracies.append(accuracy)

    if step % 10 == 0:
      t.set_description(f"loss: {float(loss_value[0]):.4f} acc: {float(accuracy):.4f}")

  avg_loss = np.mean([float(l[0]) for l in losses[-100:]])
  avg_acc = np.mean(accuracies[-100:])
  print(f"\nTraining completed. Final stats:")
  print(f"Average loss (last 100): {avg_loss:.4f}")
  print(f"Average accuracy (last 100): {avg_acc:.4f}")
  return losses, accuracies

def evaluate(model, gpu=False):
  def numpy_eval():
    X_test_tensor = Tensor(X_test.reshape((-1, 28*28)).astype(np.float32), gpu=gpu)
    Y_test_preds_out = model.forward(X_test_tensor).to_cpu()
    Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
    return (Y_test == Y_test_preds).mean()
  accuracy = numpy_eval()
  threshold = 0.88 # TODO: make this higher
  assert accuracy > threshold

class TestModels(unittest.TestCase):
  def test_conv_cpu(self):
    model = SimpleConvNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, optimizer, steps=300)
    evaluate(model)
  def test_mnist_conv_adam(self):
    model = SimpleConvNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, steps=300)
    evaluate(model)
  def test_mnist_mlp_sgd(self):
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, steps=1000)
    evaluate(model)
  def test_sgd_gpu(self):
    model = SimpleMLP()
    for param in model.parameters(): param.data = param.data * 0.1
    [x.gpu_() for x in model.parameters()]
    optimizer = optim.SGD(model.parameters(), lr=0.0002, clip_value=1.0)
    train(model, optimizer, steps=1200, BS=32, gpu=True)
    evaluate(model, gpu=True)
  def test_mnist_mlp_rmsprop(self):
    model = SimpleMLP()
    optimizer = optim.RMSprop(model.parameters(), lr=0.0002)
    train(model, optimizer, steps=1000)
    evaluate(model)

if __name__ == '__main__':
  unittest.main()
