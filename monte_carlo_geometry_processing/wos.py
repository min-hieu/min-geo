import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

MAX_WALKS = 10000
MAX_SAMPLE = 100
EPS = 1e-8
device = 'cpu'

def _length(a):
    return (a**2).sum(dim=-1).sqrt()

def circle_sdf(p, c, r):
    return r - _length(p - c)

def main_sdf(x):
    return -circle_sdf(x, 0, 0.5)

def dirichlet_bc(x, lamb=2):
    # return x[:,0]**3 + x[:,0]**2
    return torch.sin(torch.arctan2(x[:,1],x[:,0])*lamb).clip(0)

def rand_s3(n):
    theta_z = torch.rand(2,n)
    out = torch.ones(n,3)
    out[:,0] = torch.sin(theta_z[0]*np.pi*2)
    out[:,1] = torch.cos(theta_z[0]*np.pi*2)
    out[:,:2] *= ((1-(theta_z[1]*2-1)**2)**0.5)[..., None]
    out[:,2] = theta_z[1]
    return out

def wos(x, lamb=7, cache=False):
    valid_idx = torch.nonzero(main_sdf(x) < 0).squeeze()
    x_valid   = x[valid_idx.repeat(MAX_SAMPLE)]
    x_duff    = torch.cat((x_valid, torch.ones(x_valid.shape[0], 1, device=x_valid.device)), dim=-1)

    for k in range(MAX_WALKS):
        R = main_sdf(x_duff[:,:-1])
        nonterminated = torch.nonzero(R > EPS)
        if nonterminated.shape[0] == 0:
            break
        x_duff[nonterminated] += R[nonterminated][...,None] * rand_s3(nonterminated[0].shape[0]);

    if cache:
        return x_duff, valid_idx, x.shape[0]

    u_valid = dirichlet_bc(x_duff[:,:-1], lamb) * torch.cosh(lamb * x_duff[:,-1])
    u_valid = u_valid.reshape((MAX_SAMPLE,-1)).mean(dim=0)

    u = torch.zeros(x.shape[0]).double().to(x.device)
    u[valid_idx] = u_valid
    return u

def cached_wos(x_cache, valid_idx, N, lamb=2):
    u_valid = dirichlet_bc(x_cache[:,:-1], lamb) * torch.cosh(lamb * x_cache[:,-1])
    u_valid = u_valid.reshape((MAX_SAMPLE,-1)).mean(dim=0)

    u = torch.zeros(N).double().to(x_cache.device)
    u[valid_idx] = u_valid
    return u

res = 2048
# res = 1
xy = torch.tensor(np.mgrid[0:res, 0:res] / res - .5).to(device)
freqs = torch.tensor([i*3.6 for i in range(20)]).to(device)

def without_boundary_cache():
    u_spectrum = torch.zeros(len(freqs), res, res).to(device)
    for i in tqdm(range(len(freqs))):
        u_spectrum[i] = wos(xy.reshape(-1,2), freqs[i]).reshape(res, res)

    time = 2
    u_time = (u_spectrum * torch.cos(time * freqs[..., None, None])).sum(dim=0)
    # plt.matshow(u_time)
    # plt.show()

def with_boundary_cache():
    u_spectrum = torch.zeros(len(freqs), res, res).to(device)

    x_cache, valid_idx, N = wos(xy.reshape(-1,2), cache=True)
    for i in tqdm(range(len(freqs))):
        u_spectrum[i] = cached_wos(x_cache, valid_idx, N, freqs[i]).reshape(res, res)

    time = 2
    u_time = (u_spectrum * torch.cos(time * freqs[..., None, None])).sum(dim=0)
    # plt.matshow(u_time)
    # plt.show()

without_boundary_cache()
# with_boundary_cache()
