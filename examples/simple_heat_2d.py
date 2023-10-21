#!/usr/bin/env python3
#%%
import autograd.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import nlopt
from autograd import value_and_grad
from skimage import filters
from tofea.fea2d import FEA2D_T
from tofea.topopt_helpers import simp_parametrization

max_its = 500
volfrac = 0.05
sigma = 0.5
shape = (100, 100)
nelx, nely = shape
cmin, cmax = 1.0, 2e6

fixed = np.zeros((nelx + 1, nely + 1), dtype="?")
load = np.zeros_like(fixed, dtype=float)

fixed[40:60, -10:] = 1
load[20:40, 30:50] = 1
load[60:80, 30:50] = 2
component = 0

fem = FEA2D_T(fixed)
parametrization = simp_parametrization(shape, sigma, cmin, cmax)
x0 = np.full(shape, volfrac)

def temperature(x):
    x = parametrization(x)
    x, debug = fem(x, load)
    # for i in range(3):
    #     assert onp.allclose(debug[i], debug[i+1])
    return x, np.sum(debug, axis=0)


def objective(x):
    return np.sum(temperature(x)[0])*(volume(x)+volfrac*1e-3)

def volume(x):
    x = parametrization(x)
    x = (x - cmin) / (cmax - cmin)
    return np.mean(x)


plt.ion()
fig, axs = plt.subplots(1, 5, width_ratios=[3,3,1,3,1], figsize=(10,3))
im = axs[0].imshow(parametrization(x0).T, cmap="gray_r", vmin=cmin, vmax=cmax)
imT1 = axs[1].imshow(temperature(x0)[0].T, vmin=0)
plt.colorbar(imT1, cax=axs[2])
imT2 = axs[3].imshow(filters.sobel(temperature(x0)[1].T), vmin=0)
plt.colorbar(imT2, cax=axs[4])

fig.tight_layout()
#%%

def volume_constraint(x, gd):
    v, g = value_and_grad(volume)(x)
    if gd.size > 0:
        gd[:] = g
    return v - volfrac


def nlopt_obj(x, gd):
    c, dc = value_and_grad(objective)(x)

    if gd.size > 0:
        gd[:] = dc.ravel()
    T = temperature(x)
    T1 = T[0].T
    T2 = filters.sobel(T[1].T)
    im.set_data(parametrization(x).T)
    imT1.set_data(T1)
    imT1.set_clim(vmax=np.quantile(T1, 0.9))
    imT2.set_data(T2)
    imT2.set_clim(vmax=np.quantile(T2, 0.9))
    plt.pause(0.01)

    return c


opt = nlopt.opt(nlopt.LD_CCSAQ, x0.size)
#opt.add_inequality_constraint(volume_constraint, 1e-3)
opt.set_min_objective(nlopt_obj)
opt.set_lower_bounds(0)
opt.set_upper_bounds(1)
opt.set_maxeval(max_its)
opt.optimize(x0.ravel())

plt.show(block=True)
