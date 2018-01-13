#!/usr/bin/env python
# encoding: utf-8
"""
geostat.py

A python library for Geostatistics. This is mainly for people
who want to understand how it geostatistics works. It can be applied on
simple cases, but it does not contain advanced methods.

We tried to make the code as simple and clean as possible.

Most expensive operations are based on numpy and the code is reasonably fast,
however much better using other dedicated and professional software.

Created by Philippe Renard - in December 2017
Copyright (c) 2017 Philippe Renard. All rights reserved.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from scipy.special import gamma, kv


class Covariance(object):
    """
    COVARIANCE function class.

    Computes the usual covariance functions. The lag distance can be provided
    as float values or numpy arrays.

    Methods:
       constructor(rang, sill, typ="exponential", nugget=0)
       value(lag)

    Data attributes:
        rang = range of the covariance
        sill
        typ = type of covariance = "exponential", "gaussian", "spherical"
        nugget = 0 by default

    Usage:
    >>>> co = Covariance(2.0,2,typ="spherical")
    >>>> h = np.linspace(0,3,100)
    >>>> c = co(h)
    """

    def __init__(self, rang, sill, typ="exponential", nugget=0):
        self._type = typ
        self._range = rang
        self._sill = sill
        self._nugget = nugget

    def __call__(self, h):

        h = np.asarray(h)

        C = np.zeros(h.shape)
        if self._type == 'spherical':
            range_tmp = h <= self._range
            C[range_tmp] = self._sill * (1 - 3 / 2 * h[range_tmp] / self._range\
                                         + 1 / 2 * np.power(h[range_tmp] / self._range, 3))
        elif self._type == 'exponential':
            C = self._sill * np.exp(-3 * np.abs(h / self._range))
        elif self._type == 'gaussian':
            C = self._sill * np.exp(-3 * np.power(h / self._range, 2))
 #       elif self._type == 'matern':
 #           C = 2**(1-self._sill)/gamma(self._sill)*(h/self._range)**self._sill*kv(self._sill, h/self._range)
        else:
            print('Error: type of covariance not yet available: ',self._type)

        C[h == 0] += self._nugget

        return C

    def __repr__(self):

        return ("{} covariance: range = {}, sill ={}, nugget ={}".format( \
            self._type, self._range, self._sill, self._nugget))



class Variogram(object):
    """
    Variogram model function class.

    Computes the usual variogram models. The lag distance can be provided
    as float values or numpy arrays.

    Methods:
       constructor(rang, sill, typ="exponential", nugget=0)
       value(lag)

    Data attributes:
        rang = range of the covariance
        sill
        typ = type of covariance = "exponential", "gaussian", "spherical"
        nugget = 0 by default

    Usage:
    >>>> co = Variogram(2.0,2,typ="spherical")
    >>>> h = np.linspace(0,3,100)
    >>>> c = co(h)
    """

    def __init__(self, rang, sill, typ="exponential", nugget=0):
        self._type = typ
        self._range = rang
        self._sill = sill
        self._nugget = nugget

    def __call__(self, h):

        h = np.asarray(h)

        C = np.ones(h.shape) * self._sill
        if self._type == 'spherical':
            range_tmp = h <= self._range
            C[range_tmp] = self._sill * ( 3 / 2 * h[range_tmp] / self._range \
                                         - 1 / 2 * np.power(h[range_tmp] / self._range, 3))
        elif self._type == 'exponential':
            C = self._sill * (1-np.exp(-3 * np.abs(h / self._range)))
        elif self._type == 'gaussian':
            C = self._sill * (1-np.exp(-3 * np.power(h / self._range, 2)))
        elif self._type == 'linear':
            C = self._sill * h
        elif self._type == 'power':
            C = self._sill * np.power(h, self._range)
        else:
            print('Error:  variogram type: ',self._type,' not available.')
            print('        The variogram value is not computed.')
            C = np.nan

        C[ h>0 ] += self._nugget

        return C

    def __repr__(self):

        return ("{} variogram: range = {}, sill ={}, nugget ={}".format( \
            self._type, self._range, self._sill, self._nugget))

class ExperimentalVariogram(object):
    """
    Experimental Variogram class.

    Computes the experimental variogram.

    Methods:
       constructor(self, x, v, plot_variogram_cloud=False)
       value(self, hmax, lag, plot_vario=True)

    Data attributes:
        rang = range of the covariance
        sill
        typ = type of covariance = "exponential", "gaussian", "spherical"
        nugget = 0 by default

    Usage:
    >>>> co = ExperimentalVariogram(x, v, plot_variogram_cloud = False)
    """

    def __init__(self, x, v, plot_variogram_cloud=False):
        self.h = pdist(x)

        n = len(v)
        npairs = int(n * (n - 1) / 2)

        self.g = np.zeros((npairs, 1))

        c = 0
        for i in range(n):
            for j in range(i + 1, n):
                self.g[c] = 0.5 * np.power(v[i] - v[j], 2)
                c += 1

        if (plot_variogram_cloud):
            plt.plot(self.h, self.g, '.')
            plt.xlabel('lag')
            plt.ylabel('0.5 [Z(x)-Z(x+h)]^2')
            plt.title('Variogram cloud')
            plt.show()


    def __call__(self, hmax, lag, plot_vario=True):

        dlag = lag / 2
        hexp = np.arange(dlag, hmax, lag)
        gexp = np.zeros((len(hexp), 1))

        for i in range(len(hexp)):
            start = hexp[i] - dlag
            end = hexp[i] + dlag
            gexp[i] = np.mean(self.g[np.logical_and(self.h >= start, self.h <= end)])


        self.hexp = hexp
        self.gexp = gexp

        return hexp, gexp


def Q2bounds(n):
    '''
    This function computes the upper and lower bound for the 5% confidence interval of the Q2 statistics
    computed during the cross validation. This function is used by the cross_valid class.

    Source: Kitanidis (1996)

    :param n: is the number of data points
    :return: L is the lower bound and U is the upper bound
    '''
    df = n - 1
    L = chi2.ppf(0.025, df) / df
    U = chi2.ppf(0.975, df) / df
    return L, U


class CrossValid(object):
    def __init__(self, x, v):
        self.x = x
        self.v = v
        self.n = len(v)
        self.o = np.random.permutation(self.n)
        self.L, self.U = Q2bounds(self.n)

    def __call__(self, vario, makeplot=True):
        x = self.x
        v = self.v
        o = self.o
        n = self.n
        L = self.L
        U = self.U

        estkrig = np.zeros((n - 1, 1))
        varkrig = np.zeros((n - 1, 1))
        trueval = np.zeros((n - 1, 1))

        # Orthonormal residuals
        for j in range(1, n):
            ktarget = o[j]
            kdata = o[0:j]
            estkrig[j - 1], varkrig[j - 1] = \
                ordinary_kriging_variogram(x[kdata, :], v[kdata], [x[ktarget, :]], vario)
            trueval[j - 1] = v[ktarget]

        errkrig = estkrig - trueval
        ernkrig = errkrig / np.sqrt(varkrig)

        Q1 = np.mean(ernkrig)
        Q1limit = 2 / math.sqrt(n - 1)
        if (abs(Q1) <= Q1limit):
            print('Acceptable mean normalized error: Q1 =', Q1, '<= ', Q1limit)
        else:
            print('Model must be REJECTED: mean normalized error: Q1 =', Q1, '> ', Q1limit)

        Q2 = np.mean(np.power(ernkrig, 2))

        if (Q2 >= L and Q2 <= U):
            print('Acceptable mean squared normalized error: Q2 =', Q2)
            print('   within bounds LU = [', L, U, ']')
        else:
            print('Model must be REJECTED:  mean squared normalized error: Q2 =', Q2)
            print('   out of bounds LU = [', L, U, ']')

        gM = np.exp( np.mean(np.log(np.power(errkrig, 2))))
        print('Geometric mean square error =', gM)

        Cr = np.exp( np.mean(np.log(varkrig)))
        print('Geometric mean error variance =', Cr)

        if (makeplot):
            fig = plt.figure(figsize=(12,12))
            #fig.subplots_adjust(left=0.2, wspace=0.6)

            ax = fig.add_subplot(221)
            ax.plot(estkrig, trueval, 'o', estkrig, estkrig, '-')
            ax.set_xlabel('Estimation = Z*(x)')
            ax.set_ylabel('True value = Z(x)')
            ax.set_title('Cross plot Z(x) versus Z*(x)')

            ax = fig.add_subplot(222)
            ax.hist(ernkrig)
            ax.set_ylabel('Normalized error')
            ax.set_title('Histogram of normalized error')

            ax = fig.add_subplot(223)
            ernkrig.shape = (n - 1,)
            ax.scatter(x[o[1:n], 0], x[o[1:n], 1], s=20*np.abs(ernkrig), c=np.power(ernkrig, 2), alpha=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Map of normalized error')

            ax = fig.add_subplot(224)
            residual_vario = ExperimentalVariogram(x[o[1:n],:], ernkrig)
            hmax = np.max(residual_vario.h)/2
            lag = hmax/10
            he, ge = residual_vario(hmax=hmax, lag=lag)
            ax.plot(he, ge, '.-')
            ax.plot([he[0],he[-1]], [1,1], '-r')
            ax.set_ybound(lower=0)
            ax.set_xlabel('h')
            ax.set_title('Variogram of normalized error')

            plt.show()

        return Q1, Q2, gM, Cr


def ordinary_kriging_covariance(x, v, xu, covar):
    """
    Ordinary kriging - interpolates at locations xu the values v measured at locations x.

    :param x: The positions of the data points x is a matrix with n lines = n data points, each column is a coordinate
    :param v:  The measurements at the location of the data points: a vector with n lines
    :param xu: The positions where we want to get the kriging estimates: a matrix, each line is a point
    :param covar: The covariance object (see class geostat.covar to see how to create such a covariance

    :return: vo,so: The kriged value and its variance for each unknown location: 2 vectors

    Usage:
    >>>> co = covariance(2.0,2,typ="spherical")
    >>>> vo,so = ordinary_kriging(x,v,xu,covar)
    """

    # Defines the size of the problem and agglomerate the data
    n_data = len(v)
    n_unknown = len(xu)
    n_total = n_unknown + n_data
    xall = np.concatenate((xu, x))

    # Computes the covariance for all pairs of nodes
    d = squareform(pdist(xall))
    c = covar(d)

    # Assemble the Kriging matrix
    a = np.ones((n_data + 1, n_data + 1))
    a[0:n_data, 0:n_data] = c[n_unknown:n_total, n_unknown:n_total]
    a[n_data, n_data] = 0

    # Assemble the right hand side
    b = np.ones((n_data + 1, n_unknown))
    b[0:n_data, 0:n_unknown] = c[n_unknown:n_total, 0:n_unknown]

    # Solve the Kriging system
    l = np.dot(np.linalg.inv(a), b)

    # Get the Kriging weight for each data point
    w = l[0:n_data, 0:n_unknown]

    # Prepare the calculation of Kriging values
    v.shape = (n_data, 1)

    # Computes the estimated values by multiplying the kriging weights with values
    vo = np.dot(np.transpose(w), v)

    # Get the Lagrange parameter for each unknown point
    mu = l[-1, :]
    mu.shape = (n_unknown, 1)

    # Computes the kriging variance for each unknown point
    so = np.diag(np.dot(np.transpose(w), b[0:n_data, :]))
    so.shape = (n_unknown, 1)
    so = so + covar._sill - mu

    return vo, so


def ordinary_kriging_variogram(x, v, xu, vario):
    """
    Ordinary kriging - interpolates at locations xu the values v measured at locations x.

    :param x: The positions of the data points x is a matrix with n lines = n data points, each column is a coordinate
    :param v:  The measurements at the location of the data points: a vector with n lines
    :param xu: The positions where we want to get the kriging estimates: a matrix, each line is a point
    :param covar: The covariance object (see class geostat.covar to see how to create such a covariance

    :return: vo,so: The kriged value and its variance for each unknown location: 2 vectors

    Usage:
    >>>> vario = variogram(2.0,2,typ="spherical")
    >>>> vo,so = ordinary_kriging(x,v,xu,vario)
    """

    # Defines the size of the problem and agglomerate the data
    n_data = len(v)
    n_unknown = len(xu)
    n_total = n_unknown + n_data
    xall = np.concatenate((xu, x))

    # Computes the covariance for all pairs of nodes
    d = squareform(pdist(xall))
    g = -1*vario(d)

    # Assemble the Kriging matrix
    a = np.ones((n_data + 1, n_data + 1))
    a[0:n_data, 0:n_data] = g[n_unknown:n_total, n_unknown:n_total]
    a[n_data, n_data] = 0

    # Assemble the right hand side
    b = np.ones((n_data + 1, n_unknown))
    b[0:n_data, 0:n_unknown] = g[n_unknown:n_total, 0:n_unknown]

    # Solve the Kriging system
    l = np.dot(np.linalg.inv(a), b)

    # Get the Kriging weight for each data point
    w = l[0:n_data, 0:n_unknown]

    # Prepare the calculation of Kriging values
    v.shape = (n_data, 1)

    # Computes the estimated values by multiplying the kriging weights with values
    vo = np.dot(np.transpose(w), v)

    # Get the Lagrange parameter for each unknown point
    mu = l[-1, :]
    mu.shape = (n_unknown, 1)

    # Computes the kriging variance for each unknown point
    so = np.diag(np.dot(np.transpose(w), -1*b[0:n_data, :]))
    so.shape = (n_unknown, 1)
    so = so - mu

    return vo, so

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