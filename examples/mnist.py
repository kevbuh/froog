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

class SimpleMLP:
    def __init__(self):
        # 784 pixel inputs -> 128 -> 10 output
        # TODO: why down to 128? maybe because its the BS?
        self.l1 = Tensor(layer_init(784, 128))
        self.l2 = Tensor(layer_init(128, 10))

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

class SGD:
    def __init__(self, tensors, lr):
        self.tensors = tensors
        self.lr = lr
    
    def step(self):
        for t in self.tensors:
            t.data -= self.lr * t.grad

model = SimpleMLP()
optim = SGD([model.l1, model.l2], lr=0.01)

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
    model_outputs = model.forward(x)

    # ********* backward pass *********
    loss = model_outputs.mul(y).mean() # NLL loss function
    loss.backward()
    optim.step()

    pred = np.argmax(model_outputs.data, axis=1)
    accuracy = (pred == Y).mean()

    loss = loss.data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description(f"loss: {float(loss):.2f} accuracy: {float(accuracy):.2f}")

# evaluate
def numpy_eval():
  Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28))))
  Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
  return (Y_test == Y_test_preds).mean()

accuracy = numpy_eval()
print(f"test set accuracy is {numpy_eval()}")
assert accuracy > 0.95
