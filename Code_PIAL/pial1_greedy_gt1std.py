import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.sampler import RandomSampler, BatchSampler

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from deepxde import deepxde as dde
from deepxde.deepxde.nn.pytorch.deeponet import DeepONet
from utils.func import *
from utils.pdes import diffusion_reaction
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import logging
import os

# Config
n_init = 20 # number of v(x) use to pre-train the model
n_0 = 100 # number of v(x) would be selected from the training data
n_1 = 20 # number of v(x) in n0 would be used to train the model
n_2 = 200 # total v(x) would be used to train the model

batchsize = 5000 # batchsize for training, 5000 for testing the code. Defaults to 20000.
iteration = 20000 # iteration for training, 5000 for testing the code. Defaults to 30000.
shuf_dataset = True # shuffle the dataset or not

lr = 1e-3 # learning rate
decay_step = iteration // 5 # decay step, using inverse time decay 
decay_rate = 0.5

train_data = np.load("dataset/dr_train.npz")
test_data = np.load("dataset/dr_test.npz")

train_vx = torch.from_numpy(train_data["X_train0"].astype(np.float32))
train_grid = torch.from_numpy(train_data["X_train1"].astype(np.float32))
train_uxt = torch.from_numpy(train_data["y_train"].astype(np.float32))

if shuf_dataset:
    shuf_indices = torch.randperm(train_vx.shape[0], device = train_vx.device)
    train_vx = train_vx[shuf_indices]
    train_uxt = train_uxt[shuf_indices]

test_vx = torch.from_numpy(test_data["X_test0"].astype(np.float32))
test_grid = torch.from_numpy(test_data["X_test1"].astype(np.float32))
test_uxt = torch.from_numpy(test_data["y_test"].astype(np.float32))

timing = time.strftime('%m%d-%H%M%S', time.localtime()) + "-gt1std"

os.makedirs(f"runs/{timing}", exist_ok = True)
logging.basicConfig(filename = f"runs/{timing}/train.log", level = logging.INFO)

class prepareData(torch.utils.data.Dataset):
    def __init__(self, vx, grid, uxt):
        func_num = vx.shape[0]
        point_num = grid.shape[0]
        vx = vx.repeat_interleave(point_num, dim = 0)
        grid = grid.repeat(func_num, 1)
        uxt = uxt.view(-1, 1)
        self.vx = vx # 1000 x 2
        self.grid = grid # 10201 x 2
        self.uxt = uxt # 1000 x 1

    def __getitem__(self, index: int | list[int]):
        return self.vx[index], self.grid[index], self.uxt[index]

    def __len__(self):   
        return len(self.vx)

class Trainer():
    def __init__(self):
        self.net = DeepONet(layer_sizes_branch= [101, 100, 100],
                            layer_sizes_trunk= [2, 100, 100, 100],
                            activation= {"branch": F.gelu, "trunk": F.gelu},
                            kernel_initializer= "Glorot normal")
        
        self.net.apply_output_transform(dirichlet)
        self.global_step = 0
        self.tb_writer = SummaryWriter(f"runs/{timing}")
        
        self._reset_opt()
    
    def _reset_opt(self):
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda step: 1 / (1 + decay_rate * (step / decay_step)))
        
    def train(self, iterations: int, indices: list[int]):
        self.net.train()
        self.net.requires_grad_(True)
        data = prepareData(train_vx[indices], train_grid, train_uxt[indices])
        data_sampler = BatchSampler(RandomSampler(data, replacement = False, generator = torch.Generator(torch.device("cuda"))), batchsize, True)
        
        iter_data = iter(data_sampler)
        pbar = tqdm(range(iterations), desc = f"Data: {len(indices)}")
        
        for step in pbar:
            try:
                vx, grid, u_gt = data[next(iter_data)]
            except StopIteration:
                iter_data = iter(data_sampler)
                vx, grid, u_gt = data[next(iter_data)]
            inputs = (vx.cuda(), grid.requires_grad_(True).cuda())
            u_pd = self.net(inputs)
            loss_mse = F.mse_loss(u_pd, u_gt.cuda())
            #pde_norm = torch.norm(diffusion_reaction(inputs, u_pd), dim = 1) # 10, [0,1]
            #loss_pde = pde_norm.mean()
            self.opt.zero_grad()
            loss = loss_mse
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.net.parameters(), np.inf)
            self.opt.step()
            self.lr_scheduler.step()
            if self.global_step % 200 == 0:
                pbar.set_postfix(loss_mse = loss_mse.item(), grad = grad.item())
                self.tb_writer.add_scalar("loss_mse", loss_mse, self.global_step)
                #self.tb_writer.add_scalar("loss_pde", loss_pde, self.global_step)
                self.tb_writer.add_scalar("lr", self.lr_scheduler.get_last_lr()[0], self.global_step)
                self.tb_writer.add_scalar("grad", grad, self.global_step)

            dde.grad.clear()
            self.global_step += 1
        return 
    
    def test(self, indices: list[int] = None, 
             use_points: int | torch.Tensor = 10000, 
             batch_size: int = 10000, 
             use_data: str = "test"):
        """use_points: grid, (TODO) If use_points is a tensor, it will use these points to calculate the PDE residual, otherwise it will randomly generate points to calculate,"""
        if use_data == "test":
            VX, UXT = test_vx, test_uxt
        else:
            VX, UXT = train_vx, train_uxt
            
        self.net.eval()
        self.net.requires_grad_(False)
        if indices is None:
            indices = list(range(VX.shape[0]))
        if isinstance(use_points, int):
            points_num = use_points
            used_grid = torch.rand(points_num, 2, requires_grad = True)
        else:
            points_num = use_points.shape[0]
            used_grid = use_points.requires_grad_(True)
        
        used_vx = VX[indices].repeat_interleave(points_num, 0).split(batch_size, dim = 0)
        used_grid = used_grid.repeat(len(indices), 1).split(batch_size, dim = 0)
        used_uxt = UXT[indices].view(-1, 1).split(batch_size, dim = 0)
        outputs_list = []
        pde_res_list = []
        for i, (vx, grid, uxt) in enumerate(zip(used_vx, used_grid, used_uxt)):
            uxt = uxt.cuda()
            inputs = (vx.cuda(), grid.cuda())
            outputs = self.net(inputs)
            
            pde_res = diffusion_reaction(inputs, outputs).detach()
            
            outputs_list.append(outputs)
            pde_res_list.append(pde_res)
            dde.grad.clear()
            
        pred = torch.cat(outputs_list, dim = 0).view(len(indices), points_num).cpu()
        pde_res = torch.cat(pde_res_list, dim = 0).view(len(indices), points_num).cpu()
        gt = UXT[indices]
        loss_mse = F.mse_loss(pred, gt)
        l2_rel_err = torch.norm(gt - pred, dim = 1) / torch.norm(gt, dim = 1)
        return loss_mse, l2_rel_err, pde_res.abs().mean(dim = 1)

def plot_result(index):
    v_x = test_vx[index] # 101
    u_gt = test_uxt[index]

    X = test_grid.detach().cpu().numpy() # 10201, 2
    
    sensor_value = v_x[None, :].repeat_interleave(10201, 0) # 10201, 101

    u_pd = ter.net((sensor_value.cuda(), test_grid.cuda())).detach().cpu()[:,0]
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
    fig.suptitle("$u(x,t)$")
    fig.set_size_inches(4 * 3,  4)
    ax1.set_title(f"{index}_gt")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect('equal')
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$t$")
    scatter = ax1.scatter(X[:,0], X[:, 1], c=u_gt)
    plt.colorbar(scatter, ax = ax1, shrink=0.7)

    ax2.set_title(f"{index}_pd")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_aspect('equal')
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$t$")
    scatter = ax2.scatter(X[:,0], X[:, 1], c=u_pd)
    plt.colorbar(scatter, ax = ax2, shrink=0.7)

    ax3.set_title(f"{index}_delta")
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_aspect('equal')
    ax3.set_xlabel("$x$")
    ax3.set_ylabel("$t$")
    scatter = ax3.scatter(X[:,0], X[:, 1], c=u_pd - u_gt)
    plt.colorbar(scatter, ax = ax3, shrink=0.7)

    plt.tight_layout()
    img = mlp2np(fig)
    
    return img

if __name__ == "__main__":
    ter = Trainer()
    shows = range(0, 120, 10)
    num = len(shows)
    fig, axs = plt.subplots(int(np.ceil(num / 4)), 4)
    fig.suptitle("$v(x)$")
    fig.set_size_inches(4 * 4, int(np.ceil(num / 4)) * 4)
    for i, index in enumerate(shows):
        ax = axs[i // 4, i % 4]
        ax.set_title(f"{index}")
        ax.set_xlim([0, 1])
        ax.set_ylim([-5, 5])
        x = np.linspace(0, 1, 101)
        y = train_data["X_train0"][index]
        ax.scatter(x, y, s=1)
    plt.tight_layout()
    img = mlp2np(fig)
    ter.tb_writer.add_image("$v(x)$", img, dataformats = "HWC")
    
    fig, axs = plt.subplots(int(np.ceil(num / 4)), 4)
    fig.suptitle("$u(x,t)$")
    fig.set_size_inches(4 * 4, int(np.ceil(num / 4)) * 4)
    for i, index in enumerate(shows):
        ax = axs[i // 4, i % 4]
        ax.set_title(f"{index}")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        scatter = ax.scatter(train_data["X_train1"][:, 0], train_data["X_train1"][:, 1], c=train_data["y_train"][index])
        colorbar = plt.colorbar(scatter, ax = ax, shrink=0.7)

    plt.tight_layout()
    img = mlp2np(fig)
    ter.tb_writer.add_image("$u(x,t)$", img, dataformats = "HWC")
    
    del img, fig, axs, scatter, colorbar, x, y, shows, num, index, ax
    
    # pretrain
    indices = list(range(0, n_init))
    ter.train(iteration, indices)
    i = n_init
    test_mse, l2_rel, pde_res = ter.test(use_points=test_grid)
    print(l2_rel.mean(), pde_res.mean())
    img = plot_result(0)
    ter.tb_writer.add_image(f"$u(x,t)$_0", img, len(indices), dataformats = "HWC")

    while len(indices) < n_2 and i < train_vx.shape[0]:
        ter._reset_opt()
        test_mse, l2_rel, pde_res = ter.test(list(range(i, i + n_0)), use_points = train_grid, use_data = "train")
        ter.tb_writer.add_histogram("select_pde", pde_res, ter.global_step)
        
        select_indices = (pde_res > pde_res.mean() + 1 * pde_res.std()).nonzero()[:,0] + i
        print(pde_res)
        i += n_0
        indices = indices + select_indices.tolist()
        ter.train(iteration if len(indices) != n_2 else 50000, indices)
        test_mse, l2_rel, pde_res = ter.test(use_points=test_grid)
        ter.tb_writer.add_histogram("pde_res", pde_res, ter.global_step)
        ter.tb_writer.add_histogram("l2_rel", l2_rel, ter.global_step)
        img = plot_result(0)
        print(f"0_rel: {l2_rel[0].item():.3e},mse: {test_mse.mean().item():.2e}, rel_err: {l2_rel.mean().item():.2e}, pde: {pde_res.mean().item():.2e}")
        ter.tb_writer.add_image(f"$u(x,t)$_0", img, len(indices), dataformats = "HWC")
        os.makedirs(f"model/{timing}", exist_ok=True)
        torch.save(ter.net.state_dict(), f"model/{timing}/S{ter.global_step}_L{l2_rel.mean().item():.1e}.pth")
        
    ter.tb_writer.add_text("Indices", str(shuf_indices[indices]), 0)
    logging.info(f"indices: {shuf_indices[indices]}") 