import torch
import igl
from geometry import *
from tqdm import trange, tqdm
from PIL import Image
import numpy as np

# win_arr, win = init_sdl()
is_batch = True
n_lambs = 5
lambs = torch.arange(1,n_lambs+1)*30

if is_batch:
    arr = torch.zeros(n_lambs,512,512).float()
else:
    arr = torch.zeros(512,512).float()
xy  = torch.dstack(torch.meshgrid(torch.arange(512),torch.arange(512),indexing='ij'))
uv  = xy/512-0.5
u,v = uv[...,0], uv[...,1]
l   = norm(uv)

R = 0.25

def preprocess_array(a):
    f = (a*255).clamp(0,255).numpy().astype(np.uint8)
    f = np.pad(f,((0,0),(0,0),(0,1)),'constant', constant_values=255)
    return f

def rot2d(p,angle):
    c,s = torch.cos(angle), torch.sin(angle)
    rot = torch.stack((c,-s,s,c),-1).reshape(*angle.shape,2,2)

    bsize = 128
    psplit = list(p.split(bsize,dim=-2))
    rsplit = rot.split(bsize,dim=-3)
    for i,(pi,ri) in enumerate(zip(psplit,rsplit)):
        psplit[i] = (pi[...,None,:] @ ri[None,...]).squeeze()

    return torch.cat(psplit, dim=-2)

def dot(a,b):
    return (a*b).sum(-1)

def sample_boundary(o):
    phi = torch.rand(*o.shape[:-1])*torch.pi
    r = 2*R*torch.sin(phi)
    sample = r[...,None]*sincos(phi)
    sample[...,1] -= R
    return rot2d(sample, -torch.atan2(o[...,0],o[...,1]))

def sample_exterior(o):
    onorm = norm(o)
    angle = torch.arcsin(R/onorm).abs()
    sample_angle = (torch.rand(*o.shape[:-1]) - 0.5) * angle
    d = rot2d(-o/onorm[...,None],sample_angle)
    flip = torch.sign(torch.randn(*o.shape[:-1]) - 0.5)
    do = dot(d,o)
    t = do + flip*torch.sqrt(do**2-onorm**2+R**2)
    return o + t[...,None]*d

def sample_interior(o):
    d = torch.randn_like(o)
    d /= norm(d)[:,None]
    do = dot(d,o)
    o2 = dot(o,o)
    t = -do + torch.sqrt(do**2-o2+R**2)
    t = torch.where(t<0,-t,t)
    return o + t[:,None]*d

def sample_ball(n):
    d = torch.randn(n,2)
    d /= norm(d)[:,None]
    c = torch.rand(n)
    return c*d

def dirichlet_bc(p):
    theta = torch.atan2(p[:,0],p[:,1])
    return (torch.sin(theta*8)+1)/2

def wob_laplace(p, M=20):
    is_exterior = norm(p) > R
    is_interior = norm(p) < R
    p[is_interior] = sample_interior(p[is_interior])
    p[is_exterior] = sample_exterior(p[is_exterior])
    u = torch.zeros(p.shape[0])
    for i in range(M-1):
        u += ((-1)**i)*2*dirichlet_bc(p)
        p = sample_boundary(p)
    u += ((-1)**(M-1))*dirichlet_bc(p)

    return u

def neumann_bc(p,lamb):
    angle = torch.atan2(p[:,0],p[:,1])
    cond = (-torch.pi/4 <= angle) * (angle <= torch.pi/4)
    return torch.where(cond, torch.cosh(lamb*p[:,2]), 0)

def neumann_bc_batch(p,l):
    angle = torch.atan2(p[...,0],p[...,1]) # [B,N]
    cond = (-torch.pi/4 <= angle) * (angle <= torch.pi/4)
    return torch.where(cond, torch.cosh(l[:,None]*p[...,2]), 0)

def sample_exterior_cylinder(o):
    p = torch.zeros(*o.shape[:-1],o.shape[-1]+1)
    p[...,:2] = sample_exterior(o)
    sample_angle_z = (torch.rand(*o.shape[:-1])-0.5)*torch.pi/2
    p[...,-1] = torch.tan(sample_angle_z) * norm(p[...,:2])
    return p

def sample_boundary_cylinder(p):
    p[...,:2] = sample_boundary(p[...,:2])
    sample_angle_z = (torch.rand(*p.shape[:-1])-0.5)*torch.pi/2
    p[...,-1] = torch.tan(sample_angle_z) * norm(p[...,:2])
    return p

def wob_dirichlet_wave(o, M=20, lamb=3010):
    p = sample_exterior_cylinder(o) # [N,3]
    u = torch.zeros(o.shape[0])
    for i in range(M-1):
        u += ((-1)**i)*2*dirichlet_bc(p)
        p = sample_boundary_cylinder(p)
    u += ((-1)**(M-1))*dirichlet_bc(p)
    return u

def wob_neumann_wave(o, M=20, lamb=16):
    p = sample_exterior_cylinder(o) # [N,3]
    u = torch.zeros(o.shape[0])
    for i in range(M-1):
        u += ((-1)**i)*2*neumann_bc(p,lamb)
        p = sample_boundary_cylinder(p)
    u += ((-1)**(M-1))*neumann_bc(p,lamb)
    return u

def wob_neumann_wave_batch(o, M=50):
    p = sample_exterior_cylinder(o.tile(lambs.shape[0],1,1)) # [B,N,3]
    u = torch.zeros(*p.shape[:-1]) # [B,N]
    for i in range(M-1):
        u += ((-1)**i)*2*neumann_bc_batch(p,lambs) # [B,N]
        p = sample_boundary_cylinder(p) # [B,N,3]
    u += ((-1)**(M-1))*neumann_bc_batch(p,lambs) # [B,N]
    return u

def modulate(a,t):
    return (torch.sin(lambs*t)[:,None,None] * a).sum(dim=0)/t

def make_video():
    frames = []
    idx = l > R

    for t in trange(25):
        p = uv[idx].clone()
        arr[:,idx] += wob_neumann_wave_batch(p)

        a = modulate(arr,t)
        arr_p = torch.where(a > 0,a,0)
        arr_n = torch.where(a < 0,-a,0)
        zero  = torch.zeros_like(a)
        a = preprocess_array(torch.stack((arr_n,zero,arr_p),dim=-1) + 0.2)
        frames.append(Image.fromarray(a))

    frames[0].save("wave.gif", save_all=True, append_images=frames[1:], duration=160, loop=0)


if __name__ == "__main__":
    # t = 0
    # N = 1
    # idx = l > R

    # make_video(idx)
    make_video()
    exit()
    # if is_batch:
        # while True:
            # handle_sdl_event()

            # a = modulate(arr,t)
            # arr_p = torch.where(a > 0,a,0)
            # arr_n = torch.where(a < 0,-a,0)
            # zero  = torch.zeros_like(a)
            # arr_show = torch.stack((arr_n,zero,arr_p),dim=-1)/N + 0.2
            # show_array(win_arr, win, arr_show)

            # if t % 1 == 0:
                # p = uv[idx].clone()
                # arr[:,idx] += wob_neumann_wave_batch(p)
                # N += 1
            # t += 1
    # else:
        # while True:
            # handle_sdl_event()

            # a = arr
            # arr_p = torch.where(a > 0,a,0)
            # arr_n = torch.where(a < 0,-a,0)
            # zero  = torch.zeros_like(a)
            # arr_show = torch.stack((arr_n,zero,arr_p),dim=-1)/N + 0.2
            # show_array(win_arr, win, arr_show)

            # if t % 1 == 0:
                # p = uv[idx].clone()
                # arr[idx] += wob_neumann_wave(p)
                # N += 1
            # t += 1
