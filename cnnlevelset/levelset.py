import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import io, color


def imshow(img):
    plt.imshow(img, cmap='Greys_r')
    plt.show()


def norm(x):
    return np.sqrt(np.sum(np.square(x), axis=0))


def curvature(phi):
    dphi = np.gradient(phi)
    dphiy, dphix = dphi
    dphiyy, dphiyx = np.gradient(dphiy)
    dphixy, dphixx = np.gradient(dphix)

    kappa = (dphixx*dphiy**2 - 2*dphix*dphiy*dphixy + dphiyy*dphix**2) / (1. + norm(dphi)**3)

    return kappa


def stopping_fun(img):
    return 1. / (1. + norm(np.gradient(img))**2)


dt = 1.
alpha = 1.


if __name__ == '__main__':
    # Load image and convert to grayscale
    img = io.imread('data/img/twoObj.bmp')
    img = color.rgb2gray(img)

    assert len(img.shape) == 2

    # Preprocessing: mean substraction and denoising (by applying Gaussian blur)
    img = img - np.mean(img)
    img_smooth = scipy.ndimage.filters.gaussian_filter(img, 2)

    g = stopping_fun(img_smooth)
    dg = np.gradient(g)

    # Initialize surface function phi
    # 5px from image border with value 5 and -5 elsewhere
    # The 0 level curve would be a square curve
    phi = 5 * np.ones_like(img)
    phi[5:-5, 5:-5] = -5.

    for i in range(100):
        # Calculate grad phi's norm
        dphi = np.gradient(phi)

        dphiy, dphix = dphi
        dphiyy, dphiyx = np.gradient(dphiy)
        dphixy, dphixx = np.gradient(dphix)
        kappa = curvature(phi)

        dphi_norm = norm(dphi)
        kappa = curvature(phi)
        dot = dg[0]*dphi[0] + dg[1]*dphi[1]

        # Solve level set equation's PDE
        phi = phi + dt * (g*dphi_norm + g*kappa*dphi_norm + dot)

        if i % 10 == 0:
            plt.imshow(img, cmap='Greys_r')
            plt.hold(True)
            plt.contour(phi, 0, colors='r', linewidths=[3])
            plt.draw()
            plt.hold(False)
            plt.show()

            input()

