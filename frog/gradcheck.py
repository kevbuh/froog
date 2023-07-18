import numpy as np
from frog.tensor import Tensor

def mask_like(like, mask_inx, mask_value=1.0):
  mask = np.zeros_like(like).reshape(-1) # flatten
  mask[mask_inx] = mask_value            # fill 
  return mask.reshape(like.shape)

def numerical_jacobian(model, input, eps=1e-6):
    """
    https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/
    Computes :
        First-order partial derivatives using Finite-Difference Approximation with Central Difference Method (CDM)
    Params:
        model : A frog model
        input : An input
        eps   : Perturbation step
    Returns:
        NJ    : an approx. of the Jacobian
    """
    output = model.forward(input)

    ji = input.data.reshape(-1).shape[-1]  # jacobian of input
    jo = output.data.reshape(-1).shape[-1] # jacobian of output
    
    NJ = np.zeros((ji, jo)) # NJ = Numerical Jacobian
    
    for i in range(ji):
        for o in range(jo):
            eps_pertub = mask_like(input.data, i, mask_value=eps) # TODO: what is the perturbation step?
            output_perturb_add = model.forward(Tensor(input.data + eps_pertub)).data.reshape(-1)[o] # TODO: f(x + h)
            output_perturb_sub = model.forward(Tensor(input.data - eps_pertub)).data.reshape(-1)[o] # TODO: f(x - h)
            
            grad_approx = ((output_perturb_add) - (output_perturb_sub)) / (2*eps) # CDM: (f(x + h) - f(x - h)) / (2 * h)
            NJ[i,o] = grad_approx
    return NJ