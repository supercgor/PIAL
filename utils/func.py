
import numpy as np
import torch
from deepxde import deepxde as dde
from torch import nn
from torch.nn import functional as F
from .op import interp_nd
from typing import Tuple, List, Callable, Any, Union
from collections.abc import Iterable
from deepxde.deepxde.data import BatchSampler

def dirichlet(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """
    This function is to embed the dirichlet boundary.
    The dirichlet boundary is defined as:
        :math:`G(x, t)|_{x=0, x=1} = 0`
        :math:`G(x, t)|_{t=0} = 0`
    
    Args:
        inputs (np.ndarray): _description_
        outputs (np.ndarray): _description_

    Returns:
        np.ndarray: this would make the boundary condition of output automatically satisfied.
    """
    x_trunk = inputs[1] # x_trunk.shape = (t, 2)
    x, t = x_trunk[:, 0], x_trunk[:, 1] # 10201
    scale_factor = (torch.sin(torch.pi * x) * t).unsqueeze(0)
    return scale_factor * (outputs + 1)

# Adapted from https://martin-mundt.com/tensorboard-figures/
def mlp2np(fig) -> np.ndarray:
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = img / 255.0
    return img

class PDETripleCartesianProd(dde.data.Data):
    """
    Sometime we have the numerical solution of a PDE, and also we don't want the model completely physics-driven. Using this class can help us to apply both the PINN loss and data-driven loss.
    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The fierst element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train: Tuple[np.ndarray, np.ndarray], y_train: np.ndarray, X_test, y_test):
        if len(X_train[0]) * len(X_train[1]) != y_train.size:
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if len(X_test[0]) * len(X_test[1]) != y_test.size:
            raise ValueError(
                "The testing dataset does not have the format of Cartesian product."
            )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = dde.data.BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = dde.data.BatchSampler(len(X_train[1]), shuffle=True)

    def losses(self, targets, outputs, loss_fn: Union[Callable[[Any], torch.Tensor], List[Callable[[Any], torch.Tensor]]], inputs, model, aux=None):
        if not isinstance(loss_fn, Iterable):
            loss_fn = [loss_fn]
        loss_fn = list(map(lambda f: f(inputs, outputs, targets), loss_fn))
        return loss_fn

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (
            self.train_x[0][indices_branch],
            self.train_x[1][indices_trunk],
        ), self.train_y[indices_branch, indices_trunk]

    def test(self):
        return self.test_x, self.test_y

#TODO: for testing, memory leak
class PDETriple(dde.data.Data):
    """
    Sometime we have the numerical solution of a PDE, and also we don't want the model completely physics-driven. Using this class can help us to apply both the PINN loss and data-driven loss.
    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The fierst element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train: Tuple[np.ndarray, np.ndarray], y_train: np.ndarray, X_test, y_test):
        if len(X_train[0]) * len(X_train[1]) != y_train.size:
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if len(X_test[0]) * len(X_test[1]) != y_test.size:
            raise ValueError(
                "The testing dataset does not have the format of Cartesian product."
            )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = dde.data.BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = dde.data.BatchSampler(len(X_train[1]), shuffle=True)

    def losses(self, targets, outputs, loss_fn: Union[Callable[[Any], torch.Tensor], List[Callable[[Any], torch.Tensor]]], inputs, model, aux=None):
        if not isinstance(loss_fn, Iterable):
            loss_fn = [loss_fn]
        loss_fn = list(map(lambda f: f(inputs, outputs, targets), loss_fn))
        return loss_fn

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (self.train_x[0][indices_branch], self.train_x[1][indices_trunk]), self.train_y[indices_branch, indices_trunk]

    def test_next_batch(self, batch_size: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size is None:
            return self.test_x, self.test_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (self.test_x[0][indices], self.test_x[1]), self.test_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (self.test_x[0][indices_branch], self.test_x[1][indices_trunk]), self.test_y[indices_branch, indices_trunk]
    
    def test(self):
        return self.test_x, self.test_y