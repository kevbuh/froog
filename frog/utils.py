import numpy as np
from functools import lru_cache

def dense_layer(*tensor_size):
  # TODO: why dividing by sqrt?
  ret = np.random.uniform(-1., 1., size=tensor_size)/np.sqrt(np.prod(tensor_size)) # random init weights
  return ret.astype(np.float32)

def mask_like(like, mask_inx, mask_value=1.0):
  mask = np.zeros_like(like).reshape(-1) # flatten
  mask[mask_inx] = mask_value            # fill 
  return mask.reshape(like.shape)






@lru_cache
def get_im2col_index(oy, ox, cin, H, W):
  idx_channel = np.tile(np.arange(cin).repeat(H*W), oy*ox)
  # print(f"{idx_channel=}")
  # print(f"{idx_channel.shape=}")

  idx_y = np.tile(np.arange(H).repeat(W), oy*ox*cin) + np.arange(oy).repeat(ox*cin*H*W)
  # print(f"{idx_y=}")
  # print(f"{idx_y.shape=}")

  idx_x = np.tile(np.arange(W), oy*ox*cin*H) + np.tile(np.arange(ox), oy).repeat(cin*H*W)
  # print(f"{idx_x=}")
  # print(f"{idx_x.shape=}")

  OY, OX = oy+(H-1), ox+(W-1)
  idx = idx_channel * OY * OX + idx_y * OX + idx_x
  return idx

# matlab uses these to speed up convs
def im2col(x, H, W):
  bs, cin, oy, ox = x.shape[0], x.shape[1], x.shape[2]-(H-1), x.shape[3]-(W-1)

  # idx_channel, idx_y, idx_x = get_im2col_index(oy, ox, cin, H, W)
  # tx = x[:, idx_channel, idx_y, idx_x]
  idx = get_im2col_index(oy, ox, cin, H, W)
  tx = x.reshape(bs, -1)[:, idx]

  return tx.reshape(-1, cin*W*H)

def col2im(tx, H, W, OY, OX):
  oy, ox = OY-(H-1), OX-(W-1)
  bs = tx.shape[0] // (oy * ox)
  channels_in = tx.shape[1] // (H * W)
  tx = tx.reshape(bs, oy, ox, channels_in, H, W)

  x = np.zeros((bs, channels_in, OY, OX), dtype=tx.dtype)
  for Y in range(oy):
    for X in range(ox):
      x[:, :, Y:Y+H, X:X+W] += tx[:, Y, X]
  return x.reshape(bs, channels_in, OY, OX)





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

