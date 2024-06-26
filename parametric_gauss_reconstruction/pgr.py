import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import igl
from skimage import measure
from pathlib import Path

def to_unit_cube(p):
    bbmax = p.max(dim=0).values
    bbmin = p.min(dim=0).values
    s = ((bbmax - bbmin)/2).max()*3
    c = ((bbmax + bbmin)/2)
    return (p-c)/s

def dot(a,b):
    return (a*b).sum(dim=-1)

def width(x,y,w_min,w_max,nk=7):
    d = torch.cdist(x,y,p=2) # [Nx,Ny]
    i = d.topk(min(nk,x.shape[0]), dim=1, largest=False, sorted=False)[1]
    k = y[i,...] # k nearest neighbor in Y
    w = torch.linalg.norm(x[:,None,:]-k, dim=-1).mean(dim=-1)
    return w.clamp(min=w_min, max=w_max)

def linear_solve(A,b,max_iter=2000,eps=1e-10):
    x = torch.randn_like(b, requires_grad=True).float().to(b.device)
    opt = torch.optim.SGD([x], lr=1e-6, momentum=0.9)

    pbar = tqdm(range(max_iter))

    for i in pbar:
        opt.zero_grad()
        loss = (0.5*dot(x,A@x)-dot(x,b)).mean()
        loss.backward()
        opt.step()
        pbar.set_description(f'loss: {loss.item()}')

    return x.detach()

def get_A(x,y,w):
    xy = x[:,None,:]-y[None,...]     # [Nx,Ny,2]
    d = torch.linalg.norm(xy,dim=-1) # [Nx,Ny]
    mask = torch.zeros_like(xy).bool()
    mask[d < w[:,None]] = 1
    return torch.where(mask, -xy/(2*torch.pi*w[...,None,None]**2).float(),\
                             -xy/(2*torch.pi*d[...,None]**2).float()).reshape(x.shape[0],-1)

def pgr(x,q,w_min=0.000015,w_max=0.15,k=3,alpha=2.5,device='cpu',iso_offset=0.42):
    w = width(x,x,w_min,w_max,k)
    A = get_A(x,x,w).float().to(device) # [Nx,2*Nx]
    B0 = A@A.T
    B = B0 + (alpha-1)*B0.diag().diag()
    b = torch.ones(B.shape[0]).to(device) * 0.5
    xi = linear_solve(B,b)
    mu = A.T@xi

    w = width(q,x,w_min,w_max,k)
    winding = get_A(q,x,w)@mu
    isoval = torch.median(winding)+iso_offset
    return (winding - isoval).cpu().numpy()

def pgr_mu(x,w_min=0.000015,w_max=0.15,k=3,alpha=2.5,device='cpu'):
    w = width(x,x,w_min,w_max,k)
    A = get_A(x,x,w).float().to(device) # [Nx,2*Nx]
    B0 = A@A.T
    B = B0 + (alpha-1)*B0.diag().diag()
    b = torch.ones(B.shape[0]).to(device) * 0.5
    xi = linear_solve(B,b)
    mu = A.T@xi
    return mu

def get_winding(x,q,mu,w_min=0.000015,w_max=0.15,k=3):
    w = width(q,x,w_min,w_max,k)
    winding = get_A(q,x,w)@mu
    return winding

def get_largest_contours(cons):
    areas = torch.zeros(len(cons))
    for i,c in enumerate(cons):
        c_ = torch.hstack((torch.from_numpy(c),torch.zeros(c.shape[0],1)))
        areas[i] = torch.cross(c_[1:],c_[:-1])[:,2].abs().sum()
    return cons[areas.argmax()]

def extract_surface(pth,res=256,device='cpu'):
    if (Path(pth).suffix == '.ply'):
        p,_ = igl.read_triangle_mesh(pth)
    elif (Path(pth).suffix == '.npy'):
        p = np.load(pth)
    else:
        raise Exception("invalid path")
    p = torch.tensor(p[:,:2]).unique(dim=0).float()
    p = to_unit_cube(p)

    x,y = torch.meshgrid(torch.arange(res), torch.arange(res))
    q = torch.stack((x,y),dim=-1).reshape(-1,2)/res - 0.5

    s = pgr(p.to(device),q.to(device),device=device)
    s = s.reshape(res,res)
    contours = measure.find_contours(s,0.0)

    return get_largest_contours(contours)
    # return contours

def sample_points_sketch(a,N=2048):
    if type(a) == np.ndarray:
        a = torch.from_numpy(a)
    segment_length = torch.linalg.norm(a[1:] - a[:-1], dim=-1)
    segment_num_sample = (segment_length / segment_length.sum() * N).int()
    if segment_num_sample.sum() < N:
        segment_num_sample[:N-segment_num_sample.sum()] += 1
    assert segment_num_sample.sum() == N

    samples = torch.rand(N)
    samples_out = torch.zeros(N,2)
    cnt = 0
    for i, num in enumerate(segment_num_sample):
        dir = a[i+1] - a[i]
        samples_out[cnt:cnt+num] = a[i][None,:] + samples[cnt:cnt+num,None]*dir
        cnt += num

    return samples_out

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

    return samples_out, samples_normal

def get_normals(a,p,N=2048):
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

    samples_normal /= torch.linalg.norm(samples_normal,dim=-1)[:,None]
    samples_out = to_unit_cube(samples_out)

    d = torch.cdist(p,samples_out,p=2) # [Nx,Ny]
    i = d.topk(min(16,p.shape[0]), dim=1, largest=False, sorted=False)[1]
    normals = samples_normal[i,...].mean(dim=1) # k nearest neighbor in Y
    normals /= torch.linalg.norm(normals, dim=-1)[:,None]
    return normals

def extract_surface_point_and_normal(pth,res=256,eps=1e-4,device='cpu'):
    if (Path(pth).suffix == '.ply'):
        p,_ = igl.read_triangle_mesh(pth)
    elif (Path(pth).suffix == '.npy'):
        p = np.load(pth)
    else:
        raise Exception("invalid path")
    p = torch.tensor(p[:,:2]).unique(dim=0).float()
    p = to_unit_cube(p).to(device)

    x,y = torch.meshgrid(torch.arange(res), torch.arange(res))
    q = (torch.stack((x,y),dim=-1).reshape(-1,2)/res - 0.5).to(device)

    s = pgr(p,q,device=device)
    s = s.reshape(res,res)
    contour = get_largest_contours(measure.find_contours(s,0.0))
    normals = get_normals(contour, p)

    return p, normals

def visualize(pth, res=256, device='cpu'):
    import matplotlib.pyplot as plt

    p,_ = igl.read_triangle_mesh(pth)
    p = torch.tensor(p[:,:2]).unique(dim=0).float()
    p = to_unit_cube(p)

    x,y = torch.meshgrid(torch.arange(res), torch.arange(res))
    q = torch.stack((x,y),dim=-1).reshape(-1,2)/res - 0.5

    s = pgr(p.to(device),q.to(device),device=device)
    s = s.reshape(res,res)
    c = get_largest_contours(measure.find_contours(s,0.0))

    plt.matshow(s,cmap='coolwarm')
    # s = np.flip(s.reshape(128,128).T, 0) # for visualizeation
    # for c in cons:
    plt.plot(c[:,1], c[:,0], c='y')
    plt.show()

# visualize("./115430_11599f38_0000_1_pred_1_3dnormal.ply")
# visualize("./129579_144d8158_0000_0_pred_0_3dnormal.ply")
# visualize("./128814_4cb0ca05_0000_0_pred_0_3dnormal.ply")

def visualize_sample(pth):
    import matplotlib.pyplot as plt
    ours_p,_ = igl.read_triangle_mesh(pth)
    ours_p = to_unit_cube(torch.from_numpy(ours_p))
    c = extract_surface(pth)
    p = to_unit_cube(sample_points_sketch(c))
    # plt.plot(c[:,1], c[:,0], c='y')
    plt.scatter(p[:,0], p[:,1], s=1,c='r')
    plt.scatter(ours_p[:,0], ours_p[:,1],s=1, c='b')
    plt.show()

def visualize_sample_normals(pth):
    import matplotlib.pyplot as plt
    ours_p,_ = igl.read_triangle_mesh(pth)
    # ours_p = np.load(pth)
    ours_p = to_unit_cube(torch.from_numpy(ours_p))
    c,n = extract_surface_point_and_normal(pth)
    p = to_unit_cube(c)
    # plt.plot(c[:,1], c[:,0], c='y')
    plt.axes().set_aspect('equal')
    plt.axis('off')
    plt.scatter(p[:,0], p[:,1], s=1,c='r')
    plt.savefig(f'suppl/{Path(pth).stem}_pc.png',bbox_inches='tight',dpi=300)
    plt.clf()
    plt.axes().set_aspect('equal')
    plt.axis('off')
    plt.scatter(p[:,0], p[:,1], s=1,c='r')
    plt.quiver(p[:,0], p[:,1], n[:,0], n[:,1], scale=20)
    plt.savefig(f'suppl/{Path(pth).stem}_normal.png',bbox_inches='tight',dpi=300)
    plt.clf()
    # plt.scatter(ours_p[:,0], ours_p[:,1],s=1, c='b')
    # plt.save
    # plt.show()

# visualize_sample_normals("./sketch_3.npy")
# visualize_sample_normals("./cad/sketch_0.npy")
visualize_sample_normals("./129579_144d8158_0000_0_pred_0_3dnormal.ply")
visualize_sample_normals("./115430_11599f38_0000_1_pred_1_3dnormal.ply")
visualize_sample_normals("./128814_4cb0ca05_0000_0_pred_0_3dnormal.ply")
