# import sys
# sys.path.insert(0, '/home/supercgor/gitfile/Codes/Python/PIAL/deepxde')
import numpy as np
import matplotlib.pyplot as plt
from deepxde.deepxde.data import GRF
import torch
from typing import List, Callable, Any

T = 1
D = 0.01
k = 0.01
Nt = 101

def solve_ADR(xmin: float, xmax: float, tmin: float, tmax: float, k: callable, v: callable, 
              g: callable, 
              dg: callable, 
              f: callable, 
              u0: callable, 
              Nx: int, 
              Nt: int):
    """Solve 1D
    `u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t)`
    with zero boundary condition.
    """

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    k = k(x)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
    return x, t, u

def diffusion_reaction_solver(input_function: np.ndarray, 
           T: float = 1.0, 
           D: float = 0.01, 
           k: float = 0.01, 
           Nx: int = 101, 
           Nt: int = 101):
    """Compute s(x, t) over m * Nt points for a `sensor_value` of `u`. Solve 1D
    `u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t)` with zero boundary condition.
    """
    if isinstance(input_function, torch.Tensor):
        input_function = input_function.detach().cpu().numpy()
    
    x, t, u = solve_ADR(xmin = 0, xmax = 1, tmin = 0, tmax = T, 
                        k = lambda x: D * np.ones_like(x), 
                        v = lambda x: np.zeros_like(x), 
                        g = lambda u: k * u ** 2, 
                        dg = lambda u: 2 * k * u, 
                        f = lambda x, t: np.tile(input_function[:, None], (1, Nt)), 
                        u0 = lambda x: np.zeros_like(x), 
                        Nx = Nx, Nt = Nt)
    
    xt = np.asarray(np.meshgrid(x, t, indexing = "ij")).transpose([1,2,0]) # shape (2, 101, 101)
    
    return xt, u

def GRF_get(length_scale: float, Nx: int = 101) -> np.ndarray:
    space = GRF(T, length_scale = length_scale, N= 1000, interp="cubic")
    vx = space.eval_batch(space.random(1), np.linspace(0, 1, Nx)[:, None])[0]
    return vx

if __name__ == "__main__":
    vxs = []
    uxts = []
    for i in range(1000):
        print(i, end = " ")
        vxs.append(GRF_get(0.1))
        xt, uxt = diffusion_reaction_solver(vxs[-1])
        uxts.append(uxt)
    vxs = np.stack(vxs, axis = 0, dtype = np.float32)
    uxts = np.stack(uxts, axis = 0, dtype = np.float32)
    print(vxs.shape, uxts.shape, xt.shape)
    np.savez("./DF_1000test_ls0.1_101x101.npz", info = {"size": 1000, "grid": (101, 101), "grid_sample": "uniform", "length_scale": 0.1}, vxs = vxs, uxts = uxts, xt = xt)
    