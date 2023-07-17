import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, Sampler, RandomSampler, BatchSampler, SequentialSampler, DataLoader
from typing import Callable, Any, Union, Tuple, Optional, List
import deepxde.deepxde as dde
from .ADRSolver import diffusion_reaction_solver
from multiprocessing import Pool

class sampler():
    def __init__(self, dataset: Dataset, batchsize: int = 1000, sample_mode: str = "random", steps: Optional[int] = None, drop_last: bool = False):
        self.batchsize = batchsize
        self.steps = steps
        self.sample_mode = sample_mode
        self.drop_last = drop_last
        if sample_mode == "random":
            self.sampler = BatchSampler(RandomSampler(dataset), batch_size = batchsize, drop_last = drop_last)
        else:
            self.sampler = BatchSampler(SequentialSampler(dataset), batch_size = batchsize, drop_last = drop_last)
    
    def __iter__(self):
        return iter(self.sampler)
        
class prepareData(Dataset):
    def __init__(self, vx: Tensor, grid: Tensor, uxt: Tensor):
        func_num = vx.shape[0]
        point_num = grid.shape[0]
        vx = vx.repeat_interleave(point_num, dim = 0)
        grid = grid.repeat(func_num, 1)
        uxt = uxt.view(-1, 1)
        self.vx = vx # 1000 x 2
        self.grid = grid # 10201 x 2
        self.uxt = uxt # 1000 x 1

    def __getitem__(self, index: Union[int, Tuple[int]]):
        return self.vx[index], self.grid[index], self.uxt[index]

    def __len__(self):   
        return len(self.vx)

def parallel_solver(func_solver: Callable[[Any], Any], data: List[Any], num_workers: int = 6) -> List[Any]:
    """
    Solve pde in parallel.
    Actually doing the same thing as `map(func_solver, data)`

    Args:
        func_solver (Callable[[Any], Any]): solver function.
        data (List[Any]): a list of data to be solved.
        num_workers (int, optional): workers number, the more, the faster. Defaults to 6.

    Returns:
        List[Any]: _description_
    """
    with Pool(num_workers) as pool:
        result = pool.map(func_solver, data)
    return result

def get_vxs(gen_num: int, length_scale: float = 0.1, points_num: float = 101, x_max: int = 1) -> Tensor:
    """
    Generate `v(x)` for pdes.

    Args:
        gen_num (int): the number of `v(x)` to generate
        length_scale (float, optional): the length_scale used in GRF. Defaults to 0.1.
        points_num (float, optional): the points used to evaluate `v(x)`. Defaults to 101.
        x_max (int, optional): the max length of x axis. Defaults to 1.

    Returns:
        Tensor: _description_
    """
    space = dde.data.GRF(x_max, length_scale = length_scale, N= 1000, interp="cubic")
    vxs = space.eval_batch(space.random(gen_num), np.linspace(0, x_max, points_num)[:, None])
    vxs = torch.from_numpy(vxs)
    
    return vxs