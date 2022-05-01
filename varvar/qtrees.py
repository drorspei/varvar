import numpy as np
import numba
from typing import List

from varvar.predict import predict_tree


@numba.jit("UniTuple(float64, 2)(float64[:], float64[:])", nopython=True)
def single_best_multiplicative(s2, y2) -> ("log-likelihood", "s2"):
    s2tag = 0
    logs2 = 0
    for ss, yy in zip(s2, y2):
        s2tag += yy/ss
        logs2 += np.log(ss)
    ll = -len(s2)/2 * (np.log(s2tag / len(s2)) - 1 + np.log(2 * np.pi)) - logs2/2
    return ll, s2tag / len(s2)


@numba.jit(
    "Tuple((float64, float64, float64, float64, float64, unicode_type))(float64[:], float64[:], float64[:], float64, i8)",
    nopython=True
)
def quantile_single_best_split(x, s2, y2, c, min_child_size=1000) -> ("log-likelihood1", "s2_1", "log-likelihood2", "s2_2", "threshold", "deault direction"):
    best_tup = np.NINF, np.nan, np.NINF, np.nan, np.nan, "left"
    nans = np.isnan(x)
    
    inds0 = [
        np.less_equal(x, c),
        np.greater(x, c)
    ]

    inds1 = [
        ("left", [inds0[0] | nans, inds0[1]])
    ]
    
    if np.any(nans):
        inds1.append(("right", [inds0[0], inds0[1] | nans]))

    for default, inds in inds1:
        num_inds_0 = np.sum(inds[0])
        if num_inds_0 < min_child_size or (len(y2) - num_inds_0) < min_child_size:
            continue

        r1 = single_best_multiplicative(s2[inds[0]], y2[inds[0]])
        r2 = single_best_multiplicative(s2[inds[1]], y2[inds[1]])

        if r1[0] + r2[0] > best_tup[0] + best_tup[2]:
            best_tup = r1 + r2 + (c, default)
            
    return best_tup

l = numba.typed.List()
l.append(np.array([1.]))
list_of_arrays_type = numba.typeof(l)


def best_split_signature(*extra, **kwargs):
    return numba.jit(
        numba.types.Tuple((numba.types.float64, numba.types.float64, numba.types.float64, numba.types.float64, numba.types.float64, numba.types.unicode_type, numba.types.int64))(
            list_of_arrays_type, numba.types.float64[:], numba.types.float64[:], numba.types.float64[:], numba.types.int64, *extra
        ),
        nopython=True,
        **kwargs
    )


def _best_split(X, s2, y2, q=np.r_[:1:11j][1:-1], min_child_size=1000) -> ("log-likelihood1", "s2_1", "log-likelihood2", "s2_2", "threshold", "deault direction", "feature index"):
    splits = [(np.NINF, np.nan, np.NINF, np.nan, np.nan, "left")] * (len(X) * len(q))
    for i in numba.prange(len(X) * len(q)):
        f, qi = i // len(q), i % len(q)
        x = X[f]
        c = np.nanquantile(x, q[qi])
        splits[i] = quantile_single_best_split(x, s2, y2, c, min_child_size=min_child_size)

    best_i = 0
    best = splits[0][0] + splits[0][2]
    for i in range(1, len(splits)):
        if splits[i][0] + splits[i][2] > best:
            best_i = i
            best = splits[i][0] + splits[i][2]

    return splits[best_i] + (best_i // len(q),)


parallel_best_split = best_split_signature(parallel=True)(_best_split)
sequential_best_split = best_split_signature()(_best_split)


@best_split_signature(numba.types.boolean)
def best_split(X, s2, y2, q=np.r_[:1:11j][1:-1], min_child_size=1000, threads=True):
    if threads:
        return parallel_best_split(X, s2, y2, q=q, min_child_size=min_child_size)
    else:
        return sequential_best_split(X, s2, y2, q=q, min_child_size=min_child_size)

    
tree_instance = numba.typed.List()
tree_instance.append((1., 2., 'left', -1))
tree_type = numba.typeof(tree_instance)


@numba.jit(
    tree_type(list_of_arrays_type, numba.types.float64[:], numba.types.float64[:], numba.types.int64, numba.types.float64, numba.types.float64, numba.types.float64[:], numba.types.int64, numba.types.boolean),
    # parallel=True, # can't do parallel recursive functions in numba 0.54.1
    nopython=True
)
def best_tree(
    X, s2, y2, max_depth, min_gain, ll_pred=np.NINF, q=np.r_[:1:11j][1:-1], min_child_size=1000, threads=True
) -> 'List[Union[Tuple["right tree offset", "threshold", "deault direction", "feature index"), Tuple["log-likelihood", "s2", '', '')]]':
    if max_depth <= 0:
        return numba.typed.List([single_best_multiplicative(s2, y2) + ('', -1)])
    
    ll1, s2_1, ll2, s2_2, threshold, default, feature = best_split(X, s2, y2, q=q, min_child_size=min_child_size, threads=threads)
     
    if ll1 + ll2 > ll_pred + min_gain:
        lls = [ll1, ll2]
        x = X[feature]
        nans = np.isnan(x)
        
        if default == "left":
            inds0 = [
                np.less_equal(x, threshold) | np.isnan(x),
                np.greater(x, threshold)
            ]
        else:
            inds0 = [
                np.less_equal(x, threshold),
                np.greater(x, threshold) | np.isnan(x)
            ]
        
        tree_instance = numba.typed.List()
        tree_instance.append((1., 2., 'left', -1))
        left_right = [tree_instance, tree_instance]
        
        for i in numba.prange(2):
            inds = inds0[i]
            ll = lls[i]
            left_right[i] = (
                best_tree(
                    numba.typed.List([x[inds] for x in X]),
                    s2[inds], y2[inds],
                    max_depth - 1, min_gain,
                    ll,
                    q=q,
                    min_child_size=min_child_size,
                    threads=threads
                )
            )
        
        r = numba.typed.List([(len(left_right[0]) + 0., threshold, default, feature)])
        r.extend(left_right[0])
        r.extend(left_right[1])
        
        return r
    
    return numba.typed.List([single_best_multiplicative(s2, y2) + ('', -1)])


def multiplicative_variance_trees(
    X: List[np.ndarray],
    y2: np.ndarray,
    *,
    q: List[float]=np.r_[:1:11j][1:-1],
    num_trees: int,
    max_depth: int,
    min_gain: float,
    learning_rate: float=0.3,
    min_child_size: int=1000,
    threads: bool=True
):
    """
    Fit multiplicative variance trees on the feature columns in `X` with target
    `y2`:
        y^2 ~ N(0, exp( \sum_i T_i(X) )^2)
        
    The algorithm will look for splits only at the quantiles passed in q.
    It's a bit slow really.
    But it seems to perform well.
        
    Args:
        X: List of feature arrays.
        y2: Target residuals to model, assumed to already be squared.
        num_trees: Number of trees to fit, not including a `base_value` tree.
        max_depth: Maximum depth of each tree.
        min_gain: Minimum log-likelihood increase required to make a further
          partition on a leaf node of the tree.
        learning_rate: Step size shrinkage used in update to prevent overfitting.
        min_child_size: Minimum number of instances in a child.
        q: Quantiles to go over when looking for split thresholds in each
          feature.
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
    assert num_trees >= 0, "num_trees must be non-negative"
    assert learning_rate > 0, "learning rate must be positive"
    assert min_child_size >= 1, "min_child_size must be at least 1"
    assert isinstance(threads, bool), "threads must True or False"
    
    y2 = np.asanyarray(y2)
    assert np.all(y2 >= 0), "y2 must contain non-negative values"

    q = np.asanyarray(q)
    assert np.all(q > 0), "all quantiles must be positive"
    assert np.all(q < 1), "all quantiles must be less than 1"
    
    X = numba.typed.List([x for x in X])
    
    trees = [numba.typed.List([(np.nan, np.mean(y2), '', -1)])]
    s2 = np.ones_like(y2) * np.mean(y2)
    for _ in range(num_trees):
        tree = best_tree(X, s2, y2, max_depth=max_depth, min_gain=float(min_gain), ll_pred=np.NINF, q=q, min_child_size=min_child_size, threads=threads)
        for i, (ll, s2_, empty, feature) in enumerate(tree):
            if feature == -1:
                tree[i] = ll, s2_ ** learning_rate, empty, feature
        trees.append(tree)
        r = predict_tree(tree, X)
        s2 *= r
        
    ret = list(map(list, trees))
        
    return ret
