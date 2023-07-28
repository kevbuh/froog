import numpy as np
from froog.tensor import Tensor
from froog.utils import mask_like

def jacobian(model, input):
  output = model(input)

  ji = input.data.reshape(-1).shape[-1]  # jacobian of input
  jo = output.data.reshape(-1).shape[-1] # jacobian of output
  J = np.zeros((jo, ji), dtype=np.float32)

  for o in range(jo):
    o_scalar = Tensor(mask_like(output.data, o, 1.)).mul(output).sum()
    o_scalar.backward()
    for i, grad in enumerate(input.grad.reshape(-1)):
      J[o,i] = grad
  return J

def numerical_jacobian(model, input, eps = 1e-6):
#     """
#     https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/
#     Computes :
#         First-order partial derivatives using Finite-Difference Approximation with Central Difference Method (CDM)
#     Params:
#         model : A froog model
#         input : An input
#         eps   : Perturbation step
#     Returns:
#         NJ    : an approx. of the Jacobian
#     """
  output = model(input)

  ji = input.data.reshape(-1).shape[-1]
  jo = output.data.reshape(-1).shape[-1]
  NJ = np.zeros((jo, ji), dtype=np.float32)

  for i in range(ji):
    eps_perturb = mask_like(input.data, i, mask_value = eps)
    for o in range(jo):

      output_perturb_add = model(Tensor(input.data + eps_perturb)).data.reshape(-1)[o]
      output_perturb_sub = model(Tensor(input.data - eps_perturb)).data.reshape(-1)[o]

      grad_approx = ((output_perturb_add) - (output_perturb_sub)) / (2*eps) # CDM: (f(x + h) - f(x - h)) / (2 * h)

      NJ[o][i] = grad_approx
  return NJ

def gradcheck(model, input, eps = 1e-06, atol = 1e-5, rtol = 0.001):
  """
  Checks whether computed gradient is close to numerical approximation of the Jacobian
  Params:
    model       : froog model   
    eps         : eps used to see if gradient is within tolerances
    atol        : absolute tolerance
    rtol        : relative tolerance 
  Returns:
    test_passed : bool of whether the test passed
  """
  NJ = numerical_jacobian(model, input, eps)
  J = jacobian(model, input)
  return np.allclose(J, NJ, atol=atol, rtol=rtol)