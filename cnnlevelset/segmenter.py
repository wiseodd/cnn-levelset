import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import cnnlevelset.op as op
from skimage import color
from scipy.ndimage.filters import gaussian_gradient_magnitude


def default_phi(x):
    phi = np.ones_like(x)
    phi[5:-5, 5:-5] = -1.
    return phi


def phi_from_bbox(img, bbox):
    xmin, ymin, xmax, ymax = bbox
    h, w = img.shape[:2]

    xmin = int(round(xmin * w))
    xmax = int(round(xmax * w))
    ymin = int(round(ymin * h))
    ymax = int(round(ymax * h))

    phi = np.ones_like(img)
    phi[ymin:ymax, xmin:xmax] = -1

    return phi


def stopping_fun(x, alpha):
    return 1. / (1. + alpha * op.norm(op.grad(x))**2)


def levelset_segment(img, phi=None, dt=1, v=1, sigma=1, alpha=1, n_iter=1000, print_after=None):
    img_ori = img.copy()
    img = color.rgb2gray(img)

    img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)

    g = stopping_fun(img_smooth, alpha)
    dg = op.grad(g)

    plt.imshow(g, cmap='Greys_r')
    plt.show()
    print(g.min(), g.max())

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

        if print_after is not None and i % print_after == 0:
            plt.imshow(img_ori, cmap='Greys_r')
            plt.hold(True)
            plt.contour(phi, 0, colors='r', linewidths=[3])
            plt.draw()
            plt.hold(False)
            plt.show()

    plt.imshow(img_ori, cmap='Greys_r')
    plt.hold(True)
    plt.contour(phi, 0, colors='r', linewidths=[3])
    plt.draw()
    plt.hold(False)
    plt.show()
