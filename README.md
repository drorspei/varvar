# varvar
Python package to model variance in different ways

# Multiplicative variance trees and the varvar algorithm

varvar is a greedy algorithm for multiplicative variance trees.

varvar is to variance as lightgbm/xgboost/... are to expectation.

There are currently two implementations of varvar algorithms:
1. using quantile search at every split (in `varvar.qtrees`)
2. using histograms, with binning before starting (in `varvar.htrees`)

Quantile search is much slower, but can be more accurate.

This is similar to the "exact" and "hist" modes in xgboost, except our "exact"
algorithm goes over a small (exact) subset of each feature.

Both implementation modules have a `multiplicative_variance_trees` function.

Use `varvar.predict` for prediction.

The trees are returned as plain python types and can be serialized with pickle
or even as json.

Here is an example:

```
from varvar.htrees import multiplicative_variance_trees
from varvar import predict
import numpy as np

random = np.random.RandomState(1729)
n = 200000
x = random.uniform(-1000, 1000, n)
correct_threshold = 300
sigma = 1 * (x <= correct_threshold) + 30 * (x > correct_threshold)
e = sigma * random.randn(n)

trees = multiplicative_variance_trees(
    [x], e**2,
    num_trees=2, max_depth=1, min_gain=1, learning_rate=1,
)
preds = predict(trees, [x])

found_threshold = trees[1][0][1]
print(correct_threshold, found_threshold)  # 300, 295
print(np.sqrt(min(preds)), np.sqrt(max(preds)))  # 1, 30
```

## conversion to xgboost booster

You can convert multiplicative variance trees to an xgboost booster.

This allows you to use xgboost's predict function (which actually seems to be a bit slower), and more importantly to use the shap package
to interpret varvar predictions.

```
from varvar import mvt_to_xgboost
booster = mvt_to_xgboost(trees, feature_names=["f1", "f2"])
```

You need xgboost 1.6.1 or higher installed to run this code.
