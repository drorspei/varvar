"""
Implementation of hist-varvar

Histograms are used to find an approximate best split per feature.
This was first used in LightGBM, and we took ideas from the reimplementation
here:
    https://github.com/ogrisel/pygbm
    
most of which are explained in this video:
    https://www.youtube.com/watch?v=cLpIh8Aiy2w
    
Use `multiplicative_variance_trees` to train,
and `predict` to predict
    
The first import take a while since numba compiles all the functions.
On a 2020 macbookpro it take 30s to import.
"""
from functools import partial

import numba
import numpy as np
from typing import List, Optional, Tuple

from varvar.predict import predict_tree


@numba.jit("UniTuple(float64, 2)(float64[::1], float64[::1])", parallel=True, nopython=True)
def single_best_multiplicative(y2_over_s2, log_s2) -> ("log-likelihood", "s2"):
    m = np.mean(y2_over_s2)
    return -len(y2_over_s2)/2 * (np.log(m) - 1 + np.log(2 * np.pi)) - np.sum(log_s2)/2, m


@numba.jit(
    "Tuple((uint32[::1], float32[::1], float32[::1]))(int64, uint8[::1], float64[::1], float64[::1])",
    nopython=True
)
def compute_histogram(nbins, inds, y2_over_s2, log_s2):
    counts = np.zeros(nbins, np.uint32)
    y_s_sums = np.zeros(nbins, np.float32)
    log_sums = np.zeros(nbins, np.float32)
    
    n_node_samples = len(inds)
    unrolled_upper = (n_node_samples // 4) * 4
    
    # We unroll the loop to hopefully get some simd action.
    # If we ever move to float32, maybe try 8 memory loads?
    for i in range(0, unrolled_upper, 4):
        ind0 = inds[i]
        ind1 = inds[i+1]
        ind2 = inds[i+2]
        ind3 = inds[i+3]
        
        y_s_sums[ind0] += y2_over_s2[i]
        y_s_sums[ind1] += y2_over_s2[i+1]
        y_s_sums[ind2] += y2_over_s2[i+2]
        y_s_sums[ind3] += y2_over_s2[i+3]
        
        log_sums[ind0] += log_s2[i]
        log_sums[ind1] += log_s2[i+1]
        log_sums[ind2] += log_s2[i+2]
        log_sums[ind3] += log_s2[i+3]
        
        counts[ind0] += 1
        counts[ind1] += 1
        counts[ind2] += 1
        counts[ind3] += 1
    
    for i in range(unrolled_upper, n_node_samples):
        ind0 = inds[i]
        y_s_sums[ind0] += y2_over_s2[i]
        log_sums[ind0] += log_s2[i]
        counts[ind0] += 1
        
    return counts, y_s_sums, log_sums


@numba.njit("i8(f4[::1])")
def nanargmax(a):
    i = -1
    v = a[0]
    for j, aa in enumerate(a):
        if not np.isnan(aa) and (i == -1 or v < aa):
            i = j
            v = aa
    return i


@numba.jit(
    "Tuple((float64, float64, float64, float64, float64, unicode_type))(float64[::1], uint32[::1], float32[::1], float32[::1])",
    nopython=True
)
def hist_best_split(midpoints, counts, y_s_sums, log_sums) -> ("log-likelihood1", "s2_1", "log-likelihood2", "s2_2", "threshold", "deault direction"):
    """
    Return best threshold from `midpoints` such that
        feature <= `threshold`
    is the best split in the MLE sense, along with likelihood information.
    
    The first bin _must_ be negative infinity, before last bin positive
    infinity, and last bin NaN.
    """
    DEFAULT_DIRECTION_IF_NO_NANS = "left"
    
    # if there's at most one bin with instances, then return early
    if np.sum(counts > 0) <= 1:
        return np.NINF, np.nan, np.NINF, np.nan, np.inf, "left"
    
    # we need the sums up to each midpoints
    forward_counts = np.cumsum(counts[:-1])
    forward_y_s_sums = np.cumsum(y_s_sums[:-1])
    forward_log_sums = np.cumsum(log_sums[:-1])
    
    # and we need the sums from the midpoints and on
    backward_counts = forward_counts[-1] - forward_counts
    backward_y_s_sums = forward_y_s_sums[-1] - forward_y_s_sums
    backward_log_sums = forward_log_sums[-1] - forward_log_sums
    
    # add nans where should be
    if counts[-1] > 0:
        l = [
            (forward_counts + counts[-1], forward_y_s_sums + y_s_sums[-1], forward_log_sums + log_sums[-1], backward_counts, backward_y_s_sums, backward_log_sums, "left"),
            (forward_counts, forward_y_s_sums, forward_log_sums, backward_counts + counts[-1], backward_y_s_sums + y_s_sums[-1], backward_log_sums + log_sums[-1], "right"),
        ]
    else:
        l = [
            (forward_counts, forward_y_s_sums, forward_log_sums, backward_counts, backward_y_s_sums, backward_log_sums, DEFAULT_DIRECTION_IF_NO_NANS)
        ]
    # compute log-likelihoods and variances
    # ll = -counts/2 * (np.log(y_s_sums/counts) - 1 + np.log(2 * np.pi)) - np.sum(log_sums)/2
    def nandiv(x, y):
        y[y == 0] = np.nan
        return x / y
    
    def cast(x, t):
        y = np.empty(len(x), t)
        y[:] = x[:]
        return y
    
    def hist_log_likelihood(counts_, y_s_sums, log_sums):
        counts = cast(counts_, y_s_sums.dtype)
        ret = -counts/2 * (np.log(nandiv(y_s_sums, counts)) - 1 + np.log(numba.types.f4(2 * np.pi))) - log_sums/2
        return ret
    
    lls_vars = [
        (
            hist_log_likelihood(fc, fys, fls),
            nandiv(fys, cast(fc, fys.dtype)),
            hist_log_likelihood(bc, bys, bls),
            nandiv(bys, cast(bc, bys.dtype)),
            direction
        )
        for (fc, fys, fls, bc, bys, bls, direction) in l
    ]
    
    # find mle
    max_ = np.NINF, np.nan, np.NINF, np.nan, np.inf, "left"
    for ll_var in lls_vars:
        i = nanargmax(ll_var[0] + ll_var[2])
        if ll_var[0][i] + ll_var[2][i] > max_[0] + max_[2]:
            max_ = ll_var[0][i], ll_var[1][i], ll_var[2][i], ll_var[3][i], midpoints[i], ll_var[4]
            
    return max_


list_of_arrays_type = numba.types.ListType(numba.types.f8[::1])
list_of_ints_type = numba.types.ListType(numba.types.uint8[::1])

histogram_type = numba.types.Tuple((numba.types.uint32[::1], numba.types.f4[::1], numba.types.f4[::1]))
histograms_list_type = numba.types.ListType(histogram_type)


def subtract_histograms(parent, child):
    assert all(np.all(pc >= cc) for (pc, ps, pl), (cc, cs, cl) in zip(parent, child)), "wrong!"
    res = numba.typed.List([
        (pc - cc, ps - cs, pl - cl)
        for (pc, ps, pl), (cc, cs, cl) in zip(parent, child)
    ])
    return res


def all_histograms_signature(*extra, **kwargs):
    # return lambda _: _
    return numba.jit(
        histograms_list_type(
            list_of_ints_type, list_of_arrays_type, numba.types.float64[::1], numba.types.float64[::1], *extra
        ),
        nopython=True,
        **kwargs
    )


def _all_histograms(X_inds, X_midpoints, y2_over_s2, log_s2):
    histograms = [(np.empty(0, np.uint32), np.empty(0, np.float32), np.empty(0, np.float32))] * len(X_inds)

    for i in numba.prange(len(X_inds)):
        histograms[i] = compute_histogram(len(X_midpoints[i]), X_inds[i], y2_over_s2, log_s2)
        
    return numba.typed.List(histograms)


parallel_all_histograms = all_histograms_signature(parallel=True)(_all_histograms)
sequential_all_histograms = all_histograms_signature()(_all_histograms)


@all_histograms_signature(numba.types.boolean)
def all_histograms(X_inds, X_midpoints, y2_over_s2, log_s2, threads=True):
    if threads:
        return parallel_all_histograms(X_inds, X_midpoints, y2_over_s2, log_s2)
    else:
        return sequential_all_histograms(X_inds, X_midpoints, y2_over_s2, log_s2)


def best_split_signature(*extra, **kwargs):
    # return lambda _: _
    return numba.jit(
        numba.types.Tuple((numba.types.float64, numba.types.float64, numba.types.float64, numba.types.float64, numba.types.float64, numba.types.unicode_type, numba.types.int64))(
            list_of_ints_type, list_of_arrays_type, numba.types.float64[::1], numba.types.float64[::1], histograms_list_type, *extra
        ),
        nopython=True,
        **kwargs
    )


def _best_split(X_inds, X_midpoints, y2_over_s2, log_s2, histograms) -> ("log-likelihood1", "s2_1", "log-likelihood2", "s2_2", "threshold", "deault direction", "feature index"):
    splits = [(np.NINF, np.nan, np.NINF, np.nan, np.nan, "left")] * len(X_inds)

    for i in numba.prange(len(X_inds)):
        splits[i] = hist_best_split(X_midpoints[i], *histograms[i])


    best_i = 0
    best = splits[0][0] + splits[0][2]
    for i in range(1, len(splits)):
        if splits[i][0] + splits[i][2] > best:
            best_i = i
            best = splits[i][0] + splits[i][2]
    return splits[best_i] + (best_i,)

parallel_best_split = best_split_signature(parallel=True)(_best_split)
sequential_best_split = best_split_signature()(_best_split)

@best_split_signature(numba.types.boolean)
def best_split(X_inds, X_midpoints, y2_over_s2, log_s2, histograms, threads=True):
    if threads:
        return parallel_best_split(X_inds, X_midpoints, y2_over_s2, log_s2, histograms)
    else:
        return sequential_best_split(X_inds, X_midpoints, y2_over_s2, log_s2, histograms)


@numba.jit(
    list_of_ints_type(list_of_ints_type, numba.types.boolean[::1]),
    parallel=True,
    nopython=True
)
def select_inds(X_inds, inds):
    ret = numba.typed.List([np.empty(0, np.uint8)] * len(X_inds))
    for i in numba.prange(len(X_inds)):
        ret[i] = X_inds[i][inds]
    return ret


tree_instance = numba.typed.List([(1., 2., 'left', -1)])
trees_instance = numba.typed.List([tree_instance])
tree_type = numba.typeof(tree_instance)
trees_type = numba.typeof(trees_instance)

# @numba.jit(
#     tree_type(list_of_ints_type, list_of_arrays_type, numba.types.float64[::1], numba.types.float64[::1], histograms_list_type, numba.types.int64, numba.types.float64, numba.types.float64, numba.types.int64, numba.types.boolean),
#     nopython=True
# )
def best_tree(
    X_inds, X_midpoints, y2_over_s2, log_s2, histograms, max_depth, min_gain, ll_pred=np.NINF, min_child_size=1000, threads=True
) -> 'List[Union[Tuple["right tree offset", "threshold", "deault direction", "feature index"), Tuple["log-likelihood", "s2", '', '')]]':
    if max_depth <= 0:
        return numba.typed.List([single_best_multiplicative(y2_over_s2, log_s2) + ('', -1)])
    
    if not len(histograms):
        histograms = all_histograms(X_inds, X_midpoints, y2_over_s2, log_s2, threads=threads)
    ll1, s2_1, ll2, s2_2, threshold, default, feature = best_split(X_inds, X_midpoints, y2_over_s2, log_s2, histograms, threads=threads)
     
    if ll1 + ll2 > ll_pred + min_gain:
        lls = [ll1, ll2]
        x_inds = X_inds[feature]
        x_midpoints = X_midpoints[feature]

        threshold_ind = np.argmax(x_midpoints >= threshold)
        
        if default == "left":
            nans = x_inds == len(x_midpoints) - 1
            inds0 = [np.less_equal(x_inds, threshold_ind) | nans]
        else:
            inds0 = [np.less_equal(x_inds, threshold_ind)]
        inds0.append(~inds0[0])
        
        left_right = [numba.typed.List([(1., 2., 'left', -1)])] * 2
        
        ss = np.sum(inds0[0])
        
        if ss >= min_child_size and len(log_s2) - ss >= min_child_size:
            selected_inds = [
                select_inds(X_inds, inds0[0]),
                select_inds(X_inds, inds0[1])
            ]
            
            if ss <= len(log_s2) - ss:
                left_histograms = all_histograms(selected_inds[0], X_midpoints, y2_over_s2[inds0[0]], log_s2[inds0[0]], threads=threads)
                new_histograms = [left_histograms, subtract_histograms(histograms, left_histograms)]
            else:
                right_histograms = all_histograms(selected_inds[1], X_midpoints, y2_over_s2[inds0[1]], log_s2[inds0[1]], threads=threads)
                new_histograms = [subtract_histograms(histograms, right_histograms), right_histograms]
            
            for i in range(2):
                inds = inds0[i]
                ll = lls[i]
                left_right[i] = (
                    best_tree(
                        selected_inds[i],
                        X_midpoints,
                        y2_over_s2[inds], log_s2[inds],
                        new_histograms[i],
                        max_depth - 1, min_gain,
                        ll,
                        min_child_size=min_child_size,
                        threads=threads
                    )
                )

            r = numba.typed.List([(len(left_right[0]) + 0., threshold, default, feature)])
            r.extend(left_right[0])
            r.extend(left_right[1])

            return r
    
    return numba.typed.List([single_best_multiplicative(y2_over_s2, log_s2) + ('', -1)])


@numba.jit(
    numba.types.Tuple((list_of_arrays_type, list_of_ints_type))(list_of_arrays_type, numba.types.int64, numba.types.i8[:, ::1]),
    parallel=True,
    nopython=True
)
def midpoints_and_inds(X, max_bin, sample_inds):
    X_midpoints = numba.typed.List([np.array([1.])] * len(X))
    X_inds = numba.typed.List([np.array([1], np.uint8)] * len(X))
    
    for i in numba.prange(len(X)):
        x = X[i]
        sample = x[sample_inds[i]]
        midpoints = np.nanpercentile(sample, np.linspace(0, 100, max_bin + 1)[1:-1])
        midpoints = midpoints[~np.isnan(midpoints)]
        midpoints = np.unique(midpoints)
        midpoints = np.hstack((np.array([np.NINF]), midpoints, np.array([np.inf])))
        inds = np.digitize(x, midpoints)
        midpoints = np.hstack((midpoints, np.array([np.nan])))
        
        X_inds[i] = inds.astype(np.uint8)
        X_midpoints[i] = midpoints
        
    return X_midpoints, X_inds


def multiplicative_variance_trees(
    X: List[np.ndarray],
    y2: np.ndarray,
    *,
    max_bin: int=254,
    num_trees: int,
    max_depth: int,
    min_gain: float,
    learning_rate: float=0.3,
    min_child_size: int=1000,
    base_score: float=1.,
    seed: Optional[int]=None,
    threads: bool=True
) -> List[List[Tuple[float, float, str, int]]]:
    """
    Fit multiplicative variance trees on the feature columns in `X` with target
    `y2`:
        y^2 ~ N(0, exp( \sum_i T_i(X) )^2)
        
    The algorithm uses histograms, as first used in LightGBM.
    See
        https://github.com/ogrisel/pygbm
    for more details. The algorithm here is basically the same as in pygbm,
    except we're fitting variance instead of expectation.
        
    Args:
        X: List of feature arrays.
        y2: Target residuals to model, assumed to already be squared.
        num_trees: Number of trees to fit, not including a `base_value` tree.
        max_depth: Maximum depth of each tree.
        min_gain: Minimum log-likelihood increase required to make a further
          partition on a leaf node of the tree.
        learning_rate: Step size shrinkage used in update to prevent overfitting.
        min_child_size: Minimum number of instances in a child.
        base_score: The initial prediction score for all instances.
        max_bin: Maximum number of discrete bins to bucket continuous features.
          Currently only values ip to 254 are supported.
        seed: Random seed to use when sampling features to bin the data.
        threads: Whether to use threads or not. The current implementation uses
          numba's support for openmp, make sure you have it installed.
    
    Returns:
        A list trees, where each tree is a list of tuples, each representing
          a split node or a leaf node.
        Tuples have four components:
          right sub-tree offset, threshold, default direction, feature index
        where for a leaf node feature is -1 and threshold is in fact the value
        at the leaf.
        For a split node, right tree offset is the offset from the current
          index in the tree list of the node of the right sub-tree. If the
          feature at feature index is less than the threshold, we continue to
          the the next node in the tree list, otherwise we go to the right
          sub-tree. If the feature is missing (NaN) then we use the default
          direction.
    """
    assert max_bin <= 254, "max_bin must be at most 254"
    assert num_trees >= 0, "num_trees must be non-negative"
    assert learning_rate > 0, "learning rate must be positive"
    assert min_child_size >= 1, "min_split_size must be at least 1"
    assert isinstance(threads, bool), "threads must True or False"
    
    y2 = np.asanyarray(y2)
    assert np.all(y2 >= 0), "y2 must contain non-negative values"
    
    if seed is not None:
        assert isinstance(seed, int)
        
    if seed is not None:
        rs = np.random.RandomState(seed)
    else:
        rs = np.random
    sample_inds = rs.randint(0, len(X[0]), (len(X), 5000))
    
    X = numba.typed.List(X)
    X_midpoints, X_inds = midpoints_and_inds(X, max_bin, sample_inds)
            
    trees = []
    if base_score != 1.:
        trees.append(numba.typed.List([(np.nan, base_score, '', -1)]))
    
    s2 = np.ones_like(y2) * base_score
    
    y2_over_s2 = y2 / s2
    log_s2 = np.log(s2)
    histograms = all_histograms(X_inds, X_midpoints, y2_over_s2, log_s2, threads=threads)
    
    for _ in range(num_trees):
        tree = best_tree(X_inds, X_midpoints, y2_over_s2, log_s2, histograms, max_depth=max_depth, min_gain=float(min_gain), ll_pred=np.NINF, min_child_size=min_child_size, threads=threads)
        
        for i, (ll, s2_, empty, feature) in enumerate(tree):
            if feature == -1:
                tree[i] = ll, s2_ ** learning_rate, empty, feature
                
        trees.append(tree)
        r = predict_tree(tree, X)
        s2 *= r
        y2_over_s2 = y2 / s2
        log_s2 = np.log(s2)
        
    ret = list(map(list, trees))
        
    return ret
