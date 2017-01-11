import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import cnnlevelset.op as op
from skimage import color
from scipy.ndimage.filters import gaussian_gradient_magnitude

import theano as th
import theano.tensor as T


def default_phi(x):
    phi = np.ones(x.shape[:2])
    phi[5:-5, 5:-5] = -1.
    return phi


def phi_from_bbox(img, bbox):
    xmin, ymin, xmax, ymax = bbox
    h, w = img.shape[:2]

    xmin = int(round(xmin * w))
    xmax = int(round(xmax * w))
    ymin = int(round(ymin * h))
    ymax = int(round(ymax * h))

    phi = np.ones(img.shape[:2])
    phi[ymin:ymax, xmin:xmax] = -1

    return phi


def stopping_fun(x, alpha):
    return 1. / (1. + alpha * op.norm(op.grad(x))**2)


def levelset_segment(img, phi=None, dt=1, v=1, sigma=1, alpha=1, n_iter=80, print_after=None):
    img_ori = img.copy()
    img = color.rgb2gray(img)

    img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)

    g = stopping_fun(img_smooth, alpha)
    dg = op.grad(g)

    if phi is None:
        phi = default_phi(img)

    for i in range(n_iter):
        dphi = op.grad(phi)
        dphi_norm = op.norm(dphi)
        kappa = op.curvature(phi)

        smoothing = g * kappa * dphi_norm
        balloon = g * dphi_norm * v
        attachment = op.dot(dphi, dg)

        dphi_t = smoothing + balloon + attachment

        # Solve level set geodesic equation PDE
        phi = phi + dt * dphi_t

        if print_after is not None and i != 0 and i % print_after == 0:
            plt.imshow(img_ori, cmap='Greys_r')
            plt.hold(True)
            plt.contour(phi, 0, colors='r', linewidths=[3])
            plt.draw()
            plt.hold(False)
            plt.show()

    if print_after is not None:
        plt.imshow(img_ori, cmap='Greys_r')
        plt.hold(True)
        plt.contour(phi, 0, colors='r', linewidths=[3])
        plt.draw()
        plt.hold(False)
        plt.show()

    return (phi < 0)


n = 224
idx1 = th.shared(np.array(list(range(1, n)) + [n-1], dtype='int32'), name='idx1')
idx2 = th.shared(np.array([0] + list(range(0, n-1)), dtype='int32'), name='idx2')
h = th.shared(np.array([1] + (n-2) * [0.5] + [1], dtype='float32'), name='h')


def _grad(mat):
    grad_y = (mat[idx1, :] - mat[idx2, :]) * h
    grad_x = (mat[:, idx1] - mat[:, idx2]) * h

    # Ret shape: 2x224x224
    return T.stack([grad_y, grad_x], axis=0)


def _dot(x, y, axis=0):
    return T.sum(x * y, axis=axis)


def _norm(x, axis=0, squared=False):
    sq = T.sum(T.square(x), axis=axis)
    return sq if squared else T.sqrt(sq)


def _div(dfx, dfy):
    dfy_ = _grad(dfy)
    dfyy, dfyx = dfy_[0], dfy_[1]
    dfx_ = _grad(dfx)
    dfxy, dfxx = dfx_[0], dfx_[1]
    return dfxx + dfyy


def _curvature(f):
    df = _grad(f)
    dfy, dfx = df[0], df[1]
    norm = _norm(df)
    Nx = dfx / (norm + 1e-8)
    Ny = dfy / (norm + 1e-8)
    return _div(Nx, Ny)


img = T.matrix(name='img')
phi = T.matrix(name='phi')
n_iter = T.scalar(name='n_iter', dtype='int32')
dt = T.scalar(name='dt')
v = T.scalar(name='v')
alpha = T.scalar(name='alpha')

g = 1. / (1. + alpha * _norm(_grad(img), squared=True))
dg = _grad(g)


def lsm_step(phi, g, dg, dt, v):
    dphi = _grad(phi)
    dphi_norm = _norm(dphi)
    kappa = _curvature(phi)

    # Level Set evolution
    smoothing = g * kappa * dphi_norm
    balloon = g * dphi_norm * v
    attachment = _dot(dphi, dg)

    dphi_t = smoothing + balloon + attachment

    return phi + dt * dphi_t


results, updates = th.scan(fn=lsm_step,
                           outputs_info=phi,
                           non_sequences=[g, dg, dt, v],
                           n_steps=n_iter)

final_phi = results[-1]

levelset_evolution = th.function(inputs=[img, phi, dt, v, alpha, n_iter],
                                 outputs=final_phi, updates=updates,
                                 allow_input_downcast=True)


def levelset_segment_theano(img, phi=None, dt=1, v=1, sigma=1, alpha=1, n_iter=80):
    img = color.rgb2gray(img)

    img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)

    if phi is None:
        phi = default_phi(img)

    phi = levelset_evolution(img_smooth, phi, dt, v, alpha, n_iter)

    return (phi < 0)
