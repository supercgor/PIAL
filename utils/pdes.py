import torch
from torch import nn
from torch.nn import functional as F
from .op import interp_nd
from deepxde import deepxde as dde
from typing import Callable, Any, Union, Tuple, Optional, List

def diffusion_reaction(inputs: tuple[torch.Tensor, torch.Tensor], 
                       outputs: torch.Tensor, 
                       D: torch.Tensor | int = 0.01, 
                       k: torch.Tensor | int = 0.01,
                       mode = "cartesian",
                       ) -> torch.Tensor:
    """
    This pde is defined as:
        :math:`u_t = D * u_{xx} + k * u^2 + v(x)`

    Args:
        inputs (tuple[torch.Tensor, torch.Tensor]): `v` and `x`
        outputs (_type_): `u`
        D (torch.Tensor | int): hyperparmeter, can be either an int or a batched tensor (B, 1)
        k (torch.Tensor | int): hyperparmeter, can be either an int or a batched tensor (B, 1)

    Returns:
        torch.Tensor: pde residual :math:`u_t - D * u_{xx} - k * u^2 - v(x)`
    """
    
    v, x = inputs
    is_batched = v.dim() == 2

    # shape: v (B, R), x (N, d), outputs (B, N, 1)

    u_t = dde.grad.jacobian(outputs, x, i = 0, j = 1)
    u_xx = dde.grad.hessian(outputs, x, i = 0, j = 0)
    
    v_x = interp_nd(v, x[...,None,(0,)] * 2 - 1)
    
    res = u_t - D * u_xx - k * (outputs ** 2) - v_x
    
    return res 

def diffusion_reaction_nointep(inputs: tuple[torch.Tensor, torch.Tensor], 
                       outputs: torch.Tensor, 
                       D: torch.Tensor | int = 0.01, 
                       k: torch.Tensor | int = 0.01,
                       ) -> torch.Tensor:
    """
    This pde is defined as:
        :math:`u_t = D * u_{xx} + k * u^2 + v(x)`

    Args:
        inputs (tuple[torch.Tensor, torch.Tensor]): `v(x)` and `x`
        outputs (_type_): `u`
        D (torch.Tensor | int): hyperparmeter, can be either an int or a batched tensor (B, 1)
        k (torch.Tensor | int): hyperparmeter, can be either an int or a batched tensor (B, 1)

    Returns:
        torch.Tensor: pde residual :math:`u_t - D * u_{xx} - k * u^2 - v(x)`
    """
    
    v, x = inputs
    
    u_t = dde.grad.jacobian(outputs, x, i = 0, j = 1)
    u_xx = dde.grad.hessian(outputs, x, i = 0, j = 0)
        
    res = u_t - D * u_xx - k * (outputs ** 2) - v
    
    return res
    
    