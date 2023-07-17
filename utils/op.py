import torch
from torch import nn
from torch.nn import functional as F

# Adapted from https://github.com/luo3300612/grid_sample1d, support 1D, 2D, 3D grid
def grid_sample(input: torch.Tensor, 
                grid: torch.Tensor, 
                mode:str = "bilinear", 
                padding_mode:str = "zeros", 
                align_corners: bool | None = None
                ) -> torch.Tensor:
    if grid.shape[-1] == 1:
        assert mode in ["nearest", "linear"], "1D grid only support nearest and linear mode"
        input = input.unsqueeze(-1)
        grid = grid.unsqueeze(1)
        grid = torch.cat([-torch.ones_like(grid), grid], dim=-1)
        out_shape = [grid.shape[0], input.shape[1], grid.shape[2]]
        return F.grid_sample(input, grid, 
                             mode= "bilinear" if mode == "linear" else mode, 
                             padding_mode=padding_mode, 
                             align_corners=align_corners).view(*out_shape)
    else:
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def interp_nd(grid: torch.Tensor, 
              points: torch.Tensor, 
              mode: str = "linear", 
              align_corners: bool = True, 
              padding_mode: str = "border"
              ) -> torch.Tensor:
    """
    Using torch to do interpolation.

    Args:
        grid (torch.Tensor): the input function, shape `B, C, D, (H,) (W,)` for batched input, `C, D, (H,) (W,)` for unbatched input, channel dimension is optional.
        points (torch.Tensor): `B, N, dim` for batched input, `N, dim` for unbatched input. the points should be in the range of `[-1, 1]`.
        mode (str, optional): the mode for interpolation. Defaults to "linear".

    Returns:
        torch.Tensor: shape `(B,) (C,) N
    """
    interp_dim = points.shape[-1]
    is_batched = points.dim() == 3
    is_channelled = grid.dim() -1 == is_batched + interp_dim
    
    if not is_channelled:
        grid = grid.unsqueeze(int(is_batched))
        
    if not is_batched:
        grid = grid.unsqueeze(0)
        points = points.unsqueeze(0)
    
    for _ in range(interp_dim - 1):
        points = points.unsqueeze(-2)
    
    grid = grid_sample(grid, points, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    
    for _ in range(interp_dim - 1):
        grid = grid.squeeze(-2)
    
    if not is_batched:
        grid = grid.squeeze(0)
    
    if not is_channelled:
        grid = grid.squeeze(int(is_batched))
    
    return grid

if __name__ == "__main__":
    grid = torch.arange(10).float()
    grid = torch.einsum("i,j,k->ijk", grid, grid, grid)
    points = torch.rand(10, 3) * 2 -1
    grid = grid[None, None,...][:, (0,0), ...]
    points = points[None,...]
    print(interp_nd(grid, points, mode="bilinear"))