import numpy as np
from scipy.spatial.distance import pdist, squareform

def unconditionnal_lu( x, covar, nsimuls =1 ):
    """
    Unconditional_lu: generates unconditional simulations using the Cholesky decomposition (LU)

    References: Davis (1987), Alabert (1987) and Kitanidis (1996)

    :param x: numpy array containing M positions of  grid nodes (2 columns: x and y)
    :param covar: the covariance function, it's an object of covariance type (see geostat.py)
    :param nsimuls: the number of simulations to be generated

    :return: a numpy array containing M rows (the locations given in x) and nsimuls columns (the simulations)
    """

    n = x.shape[0]  # Gets the number of nodes in the grid
    d = squareform(pdist(x))  # Computes the distance matrix between all pairs of nodes
    c = covar(d)  # Computes the covariance for all pairs of nodes

    # The core of the algorithm is here
    L = np.linalg.cholesky(c)
    r = np.random.randn(n, nsimuls)
    y = np.dot(L, r)

    return y



