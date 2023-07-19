import numpy as np

def dense_layer(*tensor_size):
  # TODO: why dividing by sqrt?
  ret = np.random.uniform(-1., 1., size=tensor_size)/np.sqrt(np.prod(tensor_size)) 
  return ret.astype(np.float32)

def mask_like(like, mask_inx, mask_value=1.0):
  mask = np.zeros_like(like).reshape(-1) # flatten
  mask[mask_inx] = mask_value            # fill 
  return mask.reshape(like.shape)

def fetch_mnist():
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
    return numpy.frombuffer(gzip.decompress(dat), dtype=numpy.uint8).copy()
  
  print("loading mnist...")
  X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
  Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
  Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
  return X_train, Y_train, X_test, Y_test