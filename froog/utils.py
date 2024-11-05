#  _______  ______    _______  _______  _______ 
# |       ||    _ |  |       ||       ||       |
# |    ___||   | ||  |   _   ||   _   ||    ___|
# |   |___ |   |_||_ |  | |  ||  | |  ||   | __ 
# |    ___||    __  ||  |_|  ||  |_|  ||   ||  |
# |   |    |   |  | ||       ||       ||   |_| |
# |___|    |___|  |_||_______||_______||_______|

import numpy as np
from functools import lru_cache
import pathlib, hashlib, os, tempfile, urllib

def fetch(url):
  if url.startswith(("/", ".")): return pathlib.Path(url)
  else: fp = pathlib.Path("_cache_dir") / "froog" / "downloads" / (hashlib.md5(url.encode('utf-8')).hexdigest())
  if not fp.is_file():
    with urllib.request.urlopen(url, timeout=10) as r:
      assert r.status == 200
      total_length = int(r.headers.get('content-length', 0))
      (path := fp.parent).mkdir(parents=True, exist_ok=True)
      with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
        while chunk := r.read(16384): f.write(chunk)
        f.close()
        if (file_size:=os.stat(f.name).st_size) < total_length: raise RuntimeError(f"fetch size incomplete, {file_size} < {total_length}")
        pathlib.Path(f.name).rename(fp)
  return fp

def fetch_mnist():
  import gzip
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
  X_train = parse(fetch(f"{BASE_URL}train-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse(fetch(f"{BASE_URL}train-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
  X_test = parse(fetch(f"{BASE_URL}t10k-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse(fetch(f"{BASE_URL}t10k-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
  return X_train, Y_train, X_test, Y_test

def mask_like(like, mask_inx, mask_value=1.0):
  mask = np.zeros_like(like).reshape(-1) # flatten
  mask[mask_inx] = mask_value            # fill 
  return mask.reshape(like.shape)

@lru_cache
def get_im2col_index(oy, ox, cin, H, W):
  idx_channel = np.tile(np.arange(cin).repeat(H*W), oy*ox)
  idx_y = np.tile(np.arange(H).repeat(W), oy*ox*cin) + np.arange(oy).repeat(ox*cin*H*W)
  idx_x = np.tile(np.arange(W), oy*ox*cin*H) + np.tile(np.arange(ox), oy).repeat(cin*H*W)
  OY, OX = oy+(H-1), ox+(W-1)
  idx = idx_channel * OY * OX + idx_y * OX + idx_x
  return idx

@lru_cache
def rearrange_col2im_index(oy, ox, cin, H, W):
  idx = get_im2col_index(oy, ox, cin, H, W)
  r_idx = np.zeros((np.max(idx)+1, H*W), dtype=idx.dtype)-1
  for i,x in enumerate(idx):
    for j in range(H*W):
      if r_idx[x,j] == -1:
        r_idx[x,j] = i
        break
  return r_idx

# im2col convolution helpers
def im2col(x, H, W):
  bs, cin, oy, ox = x.shape[0], x.shape[1], x.shape[2]-(H-1), x.shape[3]-(W-1)
  idx = get_im2col_index(oy, ox, cin, H, W)
  tx = x.reshape(bs, -1)[:, idx]

  # all the time is spent here
  # np.ravel() flattens the array into a 1-dimensional shape
  tx = tx.ravel()
  return tx.reshape(-1, cin*W*H)

def col2im(tx, H, W, OY, OX):
  oy, ox = OY-(H-1), OX-(W-1)
  bs = tx.shape[0] // (oy * ox)
  channels_in = tx.shape[1] // (H * W)

  ridx = rearrange_col2im_index(oy, ox, channels_in, H, W)
  # -1 has to be 0s
  x = np.pad(tx.reshape(bs, -1), ((0,0),(0,1)))[:, ridx].sum(axis=2)
  return x.reshape(bs, channels_in, OY, OX)