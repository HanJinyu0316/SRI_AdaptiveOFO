import numpy as np

def prox_operator(u, umin, umax):
    """  
        prox(u) = min(max(u, umin), umax)
        
    Returns:
        u_proj: np.array, projected control input within [umin, umax]
    """

    return np.clip(u, umin, umax)
