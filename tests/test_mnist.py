import os
import numpy as np
from tqdm import trange
from frog.tensor import Tensor
from frog.utils import fetch_mnist, dense_layer
import frog.optim as optim
import unittest

np.random.seed(1337)

# ********* load the mnist dataset *********
X_train, Y_train, X_test, Y_test = fetch_mnist()

# ********* creating a simple mlp *********
class SimpleMLP:
  def __init__(self):
    # 784 pixel inputs -> 128 -> 10 output
    # TODO: why down to 128? maybe because its the BS?
    self.l1 = Tensor(dense_layer(784, 128))
    self.l2 = Tensor(dense_layer(128, 10))

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

class SimpleConvNet:
  def __init__(self):
    conv_size = 4 
    channels = 17
    self.c1 = Tensor(dense_layer(channels,1,conv_size,conv_size))     # (num_filters, color_channels, kernel_h, kernel_w)
    self.l1 = Tensor(dense_layer((28-conv_size+1)**2*channels, 128))  # (28-conv+1)(28-conv+1) since kernel isn't padded
    self.l2 = Tensor(dense_layer(128, 10))                            # MNIST output is 10 classes

  def forward(self, x):
    x.data = x.data.reshape((-1, 1, 28, 28))                          # get however many number of imgs in batch
    x = x.conv2d(self.c1).relu()                                      # pass through conv first
    x = x.reshape(Tensor(np.array((x.shape[0], -1))))
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()             # then go down to mlp and softmax to get probs

class TestMNIST(unittest.TestCase):
  def test_mnist(self):
    # number of samples processed before the model is updated
    def train(model, optimizer, steps, BS=128):
      # ********* training the model *********
      losses, accuracies = [], []

      for i in (t := trange(steps)):
          # X_train.shape[0] == 60,000 --> number of images in MNIST
          # this is choosing a random training image
          samp = np.random.randint(0, X_train.shape[0], size=(BS))

          # X_train[samp] is selecting a random batch of training examples
          # 28x28 pixel size of MNIST images
          # TODO: why reshaping?
          x = Tensor(X_train[samp].reshape((-1, 28*28)).astype(np.float32))
          Y = Y_train[samp]

          # 2D array where each row corresponds to an example in
          # the batch and each column corresponds to a class.
          # len(samp) is the number of examples in the batch
          y = np.zeros((len(samp),10), np.float32)

          # selects the element of y that corresponds 
          # to the true class for each example
          # NLL loss
          y[range(y.shape[0]),Y] = -10.0
          y = Tensor(y)

          # ********* foward/backward pass *********
          model_outputs = model.forward(x)

          # ********* backward pass *********
          loss = model_outputs.mul(y).mean() # TODO: what is NLL loss function?
          loss.backward()
          optimizer.step()

          pred = np.argmax(model_outputs.data, axis=1)
          accuracy = (pred == Y).mean()
        
          loss = loss.data
          losses.append(loss)
          accuracies.append(accuracy)
          t.set_description(f"loss: {float(loss[0]):.2f} accuracy: {float(accuracy):.2f}")

    def evaluate(model):
      def numpy_eval():
        Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28)).astype(np.float32)))
        Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
        return (Y_test == Y_test_preds).mean()

      accuracy = numpy_eval()
      print(f"test set accuracy: {float(accuracy):.2f}")
      assert accuracy > 0.95

    # models
    model = SimpleConvNet()
    optimizer = optim.Adam([model.c1, model.l1, model.l2], lr=0.001)
    steps = 400
    train(model, optimizer, steps)
    evaluate(model)
    
    model = SimpleMLP()
    optimizer = optim.SGD([model.l1, model.l2], lr=0.001)
    steps = 1000
    train(model, optimizer, steps)
    evaluate(model)
    
    # RMSprop
    optimizer = optim.RMSprop([model.l1, model.l2], lr=0.001)
    train(model, optimizer, steps)
    evaluate(model)


if __name__ == '__main__':
  unittest.main()
