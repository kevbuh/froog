import numpy as np
from frog.tensor import Tensor
from frog.utils import mask_like

def jacobian(model, input):
  output = model(input)

  ji = input.data.reshape(-1).shape[-1]
  jo = output.data.reshape(-1).shape[-1]
  J = np.zeros((ji, jo))

  for o in range(jo):
    o_scalar = Tensor(mask_like(output.data, o, 1.)).mul(output).sum()
    o_scalar.backward()

    for i, grad in enumerate(input.grad.reshape(-1)):
      J[i][o] = grad
  return J

# def numerical_jacobian(model, input, eps=1e-6):
#     """
#     https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/
#     Computes :
#         First-order partial derivatives using Finite-Difference Approximation with Central Difference Method (CDM)
#     Params:
#         model : A frog model
#         input : An input
#         eps   : Perturbation step
#     Returns:
#         NJ    : an approx. of the Jacobian
#     """
#     output = model.forward(input)

#     ji = input.data.reshape(-1).shape[-1]  # jacobian of input
#     jo = output.data.reshape(-1).shape[-1] # jacobian of output
    
#     NJ = np.zeros((ji, jo)) # NJ = Numerical Jacobian
    
#     for i in range(ji):
#         for o in range(jo):
#             eps_pertub = mask_like(input.data, i, mask_value=eps) # TODO: what is the perturbation step?
#             output_perturb_add = model.forward(Tensor(input.data + eps_pertub)).data.reshape(-1)[o] 
#             output_perturb_sub = model.forward(Tensor(input.data - eps_pertub)).data.reshape(-1)[o] 
            
#             grad_approx = ((output_perturb_add) - (output_perturb_sub)) / (2*eps) # CDM: (f(x + h) - f(x - h)) / (2 * h)
#             NJ[i,o] = grad_approx
#     return NJ

def numerical_jacobian(model, input, eps = 1e-6):
  output = model(input)

  ji = input.data.reshape(-1).shape[-1]
  jo = output.data.reshape(-1).shape[-1]
  NJ = np.zeros((ji, jo))

  for i in range(ji):
    for o in range(jo):

      eps_perturb = mask_like(input.data, i, mask_value = eps)
      output_perturb_add = model(Tensor(input.data + eps_perturb)).data.reshape(-1)[o]
      output_perturb_sub = model(Tensor(input.data - eps_perturb)).data.reshape(-1)[o]

      grad_approx = ((output_perturb_add) - (output_perturb_sub)) / (2*eps)

      NJ[i,o] = grad_approx
  return NJ

# def gradcheck(model, input, eps=1e-6, atol=1e-5, rtol=0.001):
#     """
#     Checks whether computed gradient is close to numerical approximation of the Jacobian
#     Params:
#       model       : frog model   
#       eps         : eps used to see if gradient is within tolerances
#       atol        : absolute tolerance
#       rtol        : relative tolerance 
#     Returns:
#       test_passed : Bool, whether the test passed
#       J           : Analytical Jacobian (from model)
#       NJ          : Finite-Difference approx. Jacobian
#     """
#     NJ = numerical_jacobian(model, input, eps)
#     output = model.forward(input)

#     ji_shape = input.data.reshape(-1).shape[-1]
#     jo_shape = output.data.reshape(-1).shape[-1]
#     J = np.zeros((ji_shape, jo_shape))

#     for o in range(jo_shape):
#       o_scalar = Tensor(mask_like(output.data, o, 1.)).mul(output).sum() # to make isolate output, and make scalar for gradient 
#       o_scalar.backward()

#       for i, grad in enumerate(input.grad.reshape(-1)):
#         J[i][o] = grad

#     test_passed = np.allclose(J, NJ, atol=atol, rtol=rtol)
#     return test_passed, J, NJ


def gradcheck(model, input, eps = 1e-06, atol = 1e-5, rtol = 0.001):
  NJ = numerical_jacobian(model, input, eps)
  J = jacobian(model, input)

  test_passed = np.allclose(J, NJ, atol=atol, rtol=rtol)
  return test_passed, J, NJ