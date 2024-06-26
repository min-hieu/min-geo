import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, default_ip=ti.i32, default_fp=ti.f32)

a = np.linspace(-1,1,512).astype(np.float32)
uv_np = np.stack(np.meshgrid(a,a), axis=-1)
uv = ti.Vector.field(n=2, dtype=ti.f32, shape=(512,512))
uv.from_numpy(uv_np)

@ti.func
def boundary_func(x: ti.type.):
    return

@ti.func
def wob():
    if

@ti.kernel
def run_all():
    for u,v in uv:

