'''
Author: Charlie [hieuristics at kaist.ac.kr]
'''
import torch
from tqdm import tqdm
import numpy as np
import trimesh
from skimage import measure
from pathlib import Path
import multiprocessing as mp
from collections import deque

def to_unit_cube(p):
    bbmax = p.max(dim=0).values
    bbmin = p.min(dim=0).values
    s = ((bbmax - bbmin)/2).max()*3
    c = ((bbmax + bbmin)/2)
    return (p-c)/s

def dot(a,b):
    return (a*b).sum(dim=-1)

def get_largest_contours(cons):
    areas = torch.zeros(len(cons))
    for i,c in enumerate(cons):
        c_ = torch.hstack((torch.from_numpy(c),torch.zeros(c.shape[0],1)))
        areas[i] = torch.cross(c_[1:],c_[:-1])[:,2].abs().sum()
    return cons[areas.argmax()]

def width(x,y,w_min,w_max,k=7):
    d = torch.cdist(x,y,p=2) # [Nx,Ny]
    i = d.topk(min(k,x.shape[0]), dim=1, largest=False, sorted=False)[1]
    k = y[i,...] # k nearest neighbor in Y
    w = torch.linalg.norm(x[:,None,:]-k, dim=-1).mean(dim=-1)
    return w.clamp(min=w_min, max=w_max)

def optimizer_update(opt, opt_schedule, loss_hist):
    cnt = 0
    if np.mean(loss_hist) < opt_schedule[cnt][0]:
        cnt += 1
        opt = opt_schedule[cnt][1]

def linear_solve(A,b,max_iter=10000,eps=1e-10):
    x = torch.rand_like(b, requires_grad=True).float().to(b.device)
    opt = torch.optim.Adam([x], lr=1e-2)

    pbar = tqdm(range(max_iter))

    loss_hist = deque(maxlen=5)
    opt_cnt = 0
    # for i in range(max_iter):
    for i in pbar:
        if np.mean(loss_hist) < 50 and opt_cnt == 0:
            opt = torch.optim.Adam([x], lr=1e-3)
            opt_cnt += 1
        if np.mean(loss_hist) < 0.05 and opt_cnt == 1:
            opt = torch.optim.SGD([x], lr=1e-10)
            opt_cnt += 1
        opt.zero_grad()
        loss = (0.5*dot(x,A@x)-dot(x,b)).mean()
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())
        assert not torch.isnan(loss)
        pbar.set_description(f'loss: {loss.item()}')

    return x.detach()

def get_A(x,y,w):
    xy = x[:,None,:]-y[None,...]     # [Nx,Ny,3]
    d = torch.linalg.norm(xy,dim=-1) # [Nx,Ny]
    mask = torch.zeros_like(xy).bool()
    mask[d < w[:,None]] = 1
    return torch.where(mask, -xy/(4*torch.pi*w[...,None,None]**3).float(),\
                             -xy/(4*torch.pi*d[...,None]**3).float()).reshape(x.shape[0],-1)

def pgr(x,q,w_min=0.0015,w_max=0.015,k=7,alpha=1.05,device='cpu',iso_offset=0.0):
    w = width(x,x,w_min,w_max,k)
    A = get_A(x,x,w).float().to(device) # [Nx,2*Nx]
    B0 = A@A.T
    B = B0 + (alpha-1)*B0.diag().diag()
    b = torch.ones(B.shape[0]).to(device) * 0.5
    print('it\'s here')
    xi = linear_solve(B,b)
    mu = A.T@xi

    with torch.no_grad():
        q_batch = q.split(65536)
        winding = torch.zeros(q.shape[0]).cpu()
        ptr = 0
        for q_ in tqdm(q_batch):
        # for q_ in q_batch:
            q_ = q_.to(device)
            w = width(q_,x,w_min,w_max,k)
            winding[ptr:ptr+len(q_)] = (get_A(q_,x,w)@mu).cpu()
            ptr += len(q_)
            torch.cuda.empty_cache()

        isoval = winding.median()
        # return (winding - isoval).numpy()
        return winding
    # return winding

def sample_points_and_normals_sketch(a,N=2048):
    if type(a) == np.ndarray:
        a = torch.from_numpy(a)
    seg_normals = (a[1:] - a[:-1]) @ torch.tensor([[0,1],[-1,0]]).double()
    seg_normals = torch.vstack((seg_normals[-1], seg_normals, seg_normals[0]))
    ver_normals = ((seg_normals[1:] + seg_normals[:-1]) / 2).float()

    segment_length = torch.linalg.norm(a[1:] - a[:-1], dim=-1)
    segment_num_sample = (segment_length / segment_length.sum() * N).int()
    if segment_num_sample.sum() < N:
        segment_num_sample[:N-segment_num_sample.sum()] += 1
    assert segment_num_sample.sum() == N

    samples = torch.rand(N)
    samples_out = torch.zeros(N,2)
    samples_normal = torch.zeros(N,2)
    cnt = 0
    for i, num in enumerate(segment_num_sample):
        dir = a[i+1] - a[i]
        samples_out[cnt:cnt+num] = a[i][None,:] + samples[cnt:cnt+num,None]*dir
        samples_normal[cnt:cnt+num] = ver_normals[i]*(1-samples[cnt:cnt+num,None])\
                                     +ver_normals[i+1]*samples[cnt:cnt+num,None]
        cnt += num

    sample_normal = samples_normal / torch.linalg.norm(samples_normal)
    return samples_out, samples_normal

def to_tensor(args, device='cpu'):
    out = []
    for a in args:
        if torch.is_tensor(a):
            out.append(a.float().to(device))
        else:
            out.append(torch.tensor(a).float().to(device))
    return out

def get_normals(p,v,vn,device):
    p,v,vn = to_tensor((p,v,vn), device=device)
    vn /= vn.norm(dim=-1)[:,None]
    v = to_unit_cube(v)

    d = torch.cdist(p,v,p=2) # [len(p),len(v)]
    i = d.topk(min(16,p.shape[0]), dim=1, largest=False, sorted=False)[1]
    n = vn[i,...].mean(dim=1) # k nearest neighbor in v
    n /= n.norm(dim=-1)[:,None]
    return n

def extract_surface_point_and_normal(pth,res=128,eps=1e-4,device='cpu'):
    # if (pth.parent/'normals.pth').exists(): return

    p = torch.load(pth).float()
    p = to_unit_cube(p).to(device)

    a = torch.arange(res).float()
    q = torch.stack(torch.meshgrid(a,a,a, indexing='ij'),dim=-1)
    q = q.reshape(-1,3)/res - 0.5

    s = pgr(p.to(device),q,device=device)
    s = s.reshape(res,res,res)
    v,f,_,_ = measure.marching_cubes(s,0.3)
    m = trimesh.Trimesh(vertices=v,faces=f)
    trimesh.repair.fix_inversion(m)
    v, vn = m.vertices, m.vertex_normals

    n = get_normals(p,v,vn,device=device).cpu()
    torch.save(n, pth.parent/'normals.pth')

    # return p, n

def extract_surface_point(pth, device='cpu'):
    p = torch.load(pth).float()[...,1:]
    p = to_unit_cube(p).to(device)
    print(p.shape)

    s = pgr(p.to(device),p.to(device),device=device)

    import polyscope as ps
    ps.init()
    pc = ps.register_point_cloud('c', p)
    pc.add_scalar_quantity('winding', s)
    ps.show()

extract_surface_point('pos_sigma.pt')
# extract_surface_point_and_normal('./iter=1499_raw.ply', device=0)
