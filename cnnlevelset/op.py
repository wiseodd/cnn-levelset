import numpy as np


def dot(x, y, axis=0):
    return np.sum(x * y, axis=axis)


def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def curvature(f):
    df = grad(f)
    dfy, dfx = df
    dfyy, dfyx = grad(dfy)
    dfxy, dfxx = grad(dfx)

    kappa = (dfxx*dfy**2 - 2*dfx*dfy*dfxy + dfyy*dfx**2) / (1. + norm(df)**3)

    return kappa
