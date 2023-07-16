import numpy as np
from tqdm import trange
from frog.tensor import Tensor

# ********* load the mnist dataset *********
def fetch(url):
    import requests, gzip, os, hashlib, numpy
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]


# ********* creating a simple mlp *********
def layer_init(in_dim,out_dim):
    # TODO: why dividing by sqrt?
    ret = np.random.uniform(-1., 1., size=(in_dim,out_dim))/np.sqrt(in_dim*out_dim) 
    return ret.astype(np.float32)

# 784 pixel inputs -> 128 -> 10 output
# TODO: why down to 128? maybe because its the BS?
l1 = Tensor(layer_init(784, 128))
# l2 = Tensor(layer_init(128, 64))
l2 = Tensor(layer_init(128, 10))

# determines step size at each iteration
lr = 0.01 

# number of samples processed before the model is updated
BS = 128 

# ********* training the model *********
losses, accuracies = [], []

for i in (t := trange(3000)):
    # X_train.shape[0] == 60,000 --> number of images in MNIST
    # this is choosing a random training image
    samp = np.random.randint(0, X_train.shape[0], size=(BS))

    # X_train[samp] is selecting a random batch of training examples
    # 28x28 pixel size of MNIST images
    # TODO: why reshaping?
    x = Tensor(X_train[samp].reshape((-1, 28*28)))
    Y = Y_train[samp]

    # 2D array where each row corresponds to an example in
    # the batch and each column corresponds to a class.
    # len(samp) is the number of examples in the batch
    y = np.zeros((len(samp),10), np.float32)

    # selects the element of y that corresponds 
    # to the true class for each example
    y[range(y.shape[0]),Y] = -1.0
    y = Tensor(y)

    # ********* foward/backward pass *********
    x = x.dot(l1) 
    x = x.relu()
    x = x_l2 = x.dot(l2)
    # x = x.relu()
    # x = x_l3 = x.dot(l3)
    x = x.logsoftmax()
    x = x.mul(y)
    x = x.mean()
    x.backward()

    # ********* backward pass *********
    loss = x.data
    pred = np.argmax(x_l2.data, axis=1)
    accuracy = (pred == Y).mean()

    # SGD
    l1.data = l1.data - lr*l1.grad
    l2.data = l2.data - lr*l2.grad
    # l3.data = l3.data - lr*l3.grad

    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description(f"loss: {float(loss):.2f} accuracy: {float(accuracy):.2f}")

# numpy forward pass
def forward(x):
  x = x.dot(l1.data)
  x = np.maximum(x, 0)
  x = x.dot(l2.data)
#   x = np.maximum(x, 0)
#   x = x.dot(l3.data)
  return x

def numpy_eval():
  Y_test_preds_out = forward(X_test.reshape((-1, 28*28)))
  Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
  return (Y_test == Y_test_preds).mean()

print(f"test set accuracy is {numpy_eval()}")