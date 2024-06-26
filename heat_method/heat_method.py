import numpy as np
import polyscope as ps
import trimesh
import igl
import matplotlib.pyplot as plt
import scipy.sparse as sp

# v,f = igl.read_triangle_mesh("./meshes/spot.obj")
v,f = igl.read_triangle_mesh("./meshes/bunny.obj")

def edge(f):
    return

def length(v,e):
    return np.linalg.norm(v[e[...,0]] - v[e[...,1]], axis=2)

def area(l):
    s = l.sum(axis=1) * 0.5
    return np.sqrt(s * (s-l[:,0]) * (s-l[:,1]) * (s-l[:,2]))

def cotweight(l,a):
    return 0.25 * (l[:,((1,2),(0,2),(0,1))].sum(axis=2) - l[:,(0,1,2)]).sum(axis=1) / a

def cotmatrix(v,f,l,a):
    w = -0.5 * cotweight(l,a)
    L = sp.


# step 1
M = igl.massmatrix(v,f,igl.MASSMATRIX_TYPE_BARYCENTRIC)
L = igl.cotmatrix(v, f)
A = igl.doublearea(v,f)
t = igl.avg_edge_length(v, f) ** 2.0
LHS = M - t*L

delta = np.zeros(v.shape[0])
heatSrcIdx = 499
delta[heatSrcIdx] = 1.0
u = sp.linalg.spsolve(LHS, delta)

# step 2
grad = igl.grad(v,f)
grad_u = (grad @ u).reshape(-1,3)
# X = stable_normalize(grad_u).reshape(-1)
X = -grad_u / np.linalg.norm(grad_u, axis=1)[:,None]
X = X.reshape(-1)

# step 3
D = -0.25*(grad.T @ sp.diags(np.tile(A,3)))
div_X = (D @ X)
phi = sp.linalg.spsolve(L, div_X)
# phi = igl.heat_geodesic(v,f,t,np.array([heatSrcIdx]))
# phi = phi - phi[heatSrcIdx]

# phi_igl = igl.heat_geodesic(v,f,t,np.array([heatSrcIdx]))
vs = np.array([heatSrcIdx])
vt = np.array(range(len(v)))
phi_igl = igl.exact_geodesic(v,f,vs,vt)
phi_igl = phi_igl - phi_igl[heatSrcIdx]

print(phi_igl[:6], phi[:6])

print(np.isnan(phi_igl).sum())
print(phi.max(), phi_igl.max())
error = np.abs((phi - phi_igl) / phi_igl.clip(1e-8))
print("error mean: ", error.mean())
print("error max: ", error.max())
print("error min: ", (error < 0.02).sum())

# visualize
ps.init()
m = ps.register_surface_mesh("bunny", v, f)
m.add_scalar_quantity("dist", phi)
m.add_scalar_quantity("dist_igl", phi_igl)
m.add_scalar_quantity("error", error)
ps.show()
