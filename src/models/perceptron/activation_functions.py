import numpy as np

## if 'der' is set to true, the activation functions are called for its derivatives to be applied for 'back propagation',
## else the activation functions are called to compute 'activation value'.

def step_function_act(z, der=False):
    '''
    The step function is challenging for gradient-based optimization (like backpropagation) since:

    Its derivative is zero almost everywhere, so it doesn't provide useful gradient information.
    The function is non-differentiable at z=0, which complicates gradient-based learning.
    '''
    if der:
        # Derivative is 0 everywhere except at the discontinuity
        return 0
    else:
        return 1 if z > 0 else 0

'''
Activation functions like sigmoid and ReLU are typically preferred, 
as they provide smooth, non-zero derivatives that facilitate effective gradient-based learning.
'''

def sigmoid_act(x, der=False):
    if der:
        f = x/(1-x)
    else :
        f = 1/(1+ np.exp(-x))
    
    return f
    
def ReLU_act(x, der=False):
    if der:
        return np.where(x > 0, 1, 0)
    else:
        return np.maximum(0, x)
