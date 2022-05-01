"""
This is an implementation of parameter estimation for linear variance.
It requires scipy.
"""
from numba import njit
import numpy as np
import scipy.optimize 


@njit
def linear_std_f_jac(a, x, e2):
    """
    Compute the minus log-likelihood of $e^2$ given the model $e~N(0, <x,a>^2)$
    (up to a fixed additive constant) and its Jacobian with respect to `a`.
    """
    dots = 1/np.dot(x, a)
    e2dots2 = e2 * dots**2
    f = np.sum(e2dots2 / 2 - np.log(dots))
    j = np.sum(x * (dots - e2dots2 * dots).reshape(-1, 1), axis=0)
    return f, j


def mle_linear(x, *, e=None, e2=None, a0=None, bounds=None, **kwargs):
    """
    Find the MLE for `a` given the model $e~N(0, <x,a>^2)$
     
    If you pass `e2`, the squared errors, then they won't be recomputed.
    If you don't pass `e2`, then you must pass `e`.
     
    `a0`, `bounds` and `kwargs` are passed to `scipy.optimize.minimize`.
    """
    x = np.asanyarray(x)
    assert x.ndim == 2
    n, l = x.shape
    if a0 is None:
        a0 = np.ones(l)
    else:
        a0 = np.asanyarray(a0)
    if e2 is None:
        assert e is not None
        e = np.asanyarray(e)
        assert e.ndim == 1
        e2 = e**2
    else:
        e2 = np.asanyarray(e2)
        assert e2.ndim == 1
    if bounds is None:
        bounds=[(0, None)] * l
    return scipy.optimize.minimize(
        linear_std_f_jac, a0,
        args=(x, e2), jac=True, bounds=bounds, **kwargs
    )

