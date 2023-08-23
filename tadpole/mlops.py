from froog.tensor import Function

class Contiguous(Function):
  pass

class Cast(Function):
  pass

# ---------Unary---------

class Sin(Function):
  pass

class Relu(Function):
  pass

class Log(Function):
  pass

class Exp(Function):
  pass

class Sqrt(Function):
  pass

class Sigmoid(Function):
  pass

# ---------Reduce---------

class Sum(Function):
  pass

class Max(Function):
  pass

# ---------Binary---------

class Less(Function):
  pass

class Add(Function):
  pass

class Sub(Function):
  pass

class Mul(Function):
  pass

class Div(Function):
  pass

# ---------Ternary---------

class Where(Function):
  pass

# ---------Movement---------

class Expand(Function):
  pass

class Reshape(Function):
  pass

class Permute(Function):
  pass

class Pad(Function):
  pass

class Shrink(Function):
  pass

class Flip(Function):
  pass