import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import cnnlevelset.op as op
from skimage import color


def default_phi(x):
    phi = 5 * np.ones_like(x)
    phi[5:-5, 5:-5] = -5.
    return phi


def stopping_fun(x):
    return 1. / (1. + op.norm(op.grad(x))**2)


def levelset_segment(img, phi=None, g_fun=stopping_fun, dt=1, n_iter=100, print_after=10):
    img = color.rgb2gray(img)

    # Preprocessing: mean substraction and denoising
    img = img - np.mean(img)
    img_smooth = scipy.ndimage.filters.gaussian_filter(img, 2)

    g = g_fun(img_smooth)
    dg = op.grad(g)

    if phi is None:
        phi = default_phi(img)

    for i in range(100):
        dphi = op.grad(phi)
        dphi_norm = op.norm(dphi)
        kappa = op.curvature(phi)

        # Solve level set geodesic equation PDE
        phi = phi + dt * (g*dphi_norm + g*kappa*dphi_norm + op.dot(dg, dphi))

        if i % 10 == 0:
            plt.imshow(img, cmap='Greys_r')
            plt.hold(True)
            plt.contour(phi, 0, colors='r', linewidths=[3])
            plt.draw()
            plt.hold(False)
            plt.show()

            input()
