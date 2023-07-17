import torch
import torch.nn as nn
import torch.nn.functional as F
from .op import interp_nd
from typing import Callable, Any, Union, Tuple, Optional, List

class dataLoss():
    """
    This type of loss is used to calculate the loss based on labels and outputs.
    """
    def __init__(self, func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss):
        self.loss = func
        
    def __call__(self, inputs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor, *args, **kwargs):
        return self.loss(outputs, targets, *args, **kwargs)
    
class pinnLoss():
    """
    This type of loss is used to calculate the loss based on inputs and outputs.
    """
    def __init__(self, 
                 pde: Callable[[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]], torch.Tensor], 
                 func: Callable[[torch.Tensor], torch.Tensor] = F.mse_loss, 
                 mode: str = "cartesian", 
                 v_space: Tuple[int] | None = (0,)):
        """
        Args:
            pde (_type_): _description_
            func (_type_, optional): _description_. Defaults to F.mse_loss.
            mode (str): `cartesian` or `normal`. Defaults to "cartesian".
            v_space (Tuple[int], optional):  The coordinations used in interp. If None, there will be no interp. Defaults to (0,).
        """
        self.pde = pde
        self.mode = mode
        self.loss = func
        self.v_space = v_space
    
    def __call__(self, 
                 inputs: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], 
                 outputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        
        if self.mode == "cartesian": # vs: B, R, 1, xs: N, d, outputs: B, N, 1
            vs, xs = inputs
            if self.v_space is not None:
                vs = interp_nd(vs, xs[...,self.v_space])
            outputs = outputs.unsqueeze(-1) # (B, N, 1)
            vs = vs.unsqueeze(-1) # (B, R, 1)
            values = list(map(lambda y, v: self.pde((v, xs), y), outputs, vs))
            values = torch.stack(values)
        else:
            values = self.pde(inputs, outputs)
            
        return self.loss(values, torch.zeros_like(values))