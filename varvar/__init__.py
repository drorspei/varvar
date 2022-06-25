import threading
from typing import List, Tuple, Optional

import numpy as np

from varvar.conversions import mvt_to_xgboost


def htrain(
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
    from varvar.htrees import multiplicative_variance_trees as _htrain
    return _htrain(
        X=X,
        y2=y2,
        max_bin=max_bin,
        num_trees=num_trees,
        max_depth=max_depth,
        min_gain=min_gain,
        learning_rate=learning_rate,
        min_child_size=min_child_size,
        base_score=base_score,
        seed=seed,
        threads=threads
    )

def qtrain(
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
    from varvar.qtrees import multiplicative_variance_trees as _qtrain
    return _qtrain(
        X=X,
        y2=y2,
        q=q,
        num_trees=num_trees,
        max_depth=max_depth,
        min_gain=min_gain,
        learning_rate=learning_rate,
        min_child_size=min_child_size,
        threads=threads
    )

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
    from varvar.predicttrees import predict as _predict
    return _predict(trees, X)

def import_():
    import varvar.qtrees
    import varvar.htrees
    import varvar.predicttrees

_thread = threading.Thread(target=import_)
_thread.start()
