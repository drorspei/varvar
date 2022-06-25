from typing import List, Tuple

import numba
import numpy as np


list_of_arrays_type = numba.types.ListType(numba.types.f8[::1])
tree_instance = numba.typed.List([(1., 2., 'left', -1)])
trees_instance = numba.typed.List([tree_instance])
tree_type = numba.typeof(tree_instance)
trees_type = numba.typeof(trees_instance)


@numba.jit(
    numba.types.void(tree_type, numba.types.int64, list_of_arrays_type, numba.types.float64[::1], numba.types.int64[::1]),
    nopython=True
)
def predict_tree_inplace(tree, offset, X, y, inds):
    if not X:
        return
    if tree[offset][3] == -1:
        y[inds] *= tree[offset][1]
    else:
        right_tree_offset, threshold, default, feature = tree[offset]
        right_tree_offset = int(right_tree_offset)
        x = X[feature][inds]
        nans = np.isnan(x)
        
        inds0 = [
            np.less_equal(x, threshold),
            np.greater(x, threshold)
        ]

        inds0[0 if default == "left" else 1] |= nans
        offsets = [0, right_tree_offset]
        for i in numba.prange(2):
            predict_tree_inplace(tree, offset + 1 + offsets[i], X, y, inds[inds0[i]])


@numba.jit(
    numba.types.float64[::1](tree_type, list_of_arrays_type),
    nopython=True
)
def predict_tree(tree, X):
    if not X:
        return np.zeros(0)
    X = numba.typed.List(X)
    y = np.ones(len(X[0]))
    predict_tree_inplace(tree, 0, X, y, np.arange(len(y)))
    return y


@numba.jit(
    numba.types.int64[::1](numba.types.i8, numba.types.i8),
    nopython=True
)
def div_points(Ntotal, Nsections):
    Neach_section, extras = divmod(Ntotal, Nsections)
    section_sizes = ([0] +
                     [Neach_section+1] * extras +
                     [Neach_section] * (Nsections-extras))
    ret = np.cumsum(np.array(section_sizes, dtype=np.int64))
        
    return ret


@numba.jit(
    numba.types.float64[::1](trees_type, list_of_arrays_type),
    parallel=True,
    nopython=True
)
def _predict(trees, X):
    n = numba.get_num_threads()
    r = [np.array([1.])[:0]] * n
    points = div_points(len(trees), n)
    
    for i in numba.prange(n):
        st = points[i]
        end = points[i + 1]
        if st < end:
            a = predict_tree(trees[st], X)
            for t in trees[st + 1:end]:
                a *= predict_tree(t, X)
            r[i] = a
        
    res = np.ones(len(r[0]))
    points = div_points(len(r[0]), n)
    
    for i in numba.prange(n):
        st = points[i]
        end = points[i + 1]
        if st < end:
            for rr in r:
                if len(rr):
                    res[st:end] *= rr[st:end]
        
    return res


def predict(
    trees: List[List[Tuple[float, float, str, int]]],
    X: List[np.ndarray]
) -> np.ndarray:
    """
    Predicts the variance of samples using the given multiplicative
    variance trees and features.
    
    The structure of `trees` is explained in the return value of
    `multiplicative_variance_trees`.
    
    Args:
        trees: As returned from `multiplicative_variance_trees`.
        X: A list of feature arrays, in the same semantic ordering as in the
          training stage.
          
    Returns:
        Numpy array of positive floats, representing predictions of the
        variance of the random variable conditioned on the features.
    """
    return _predict(
        numba.typed.List(map(numba.typed.List, trees)),
        numba.typed.List(X)
    )
