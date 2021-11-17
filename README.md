# varvar
Python package to model variance in different ways

# Multiplicative variance trees

Like additive regression trees.

There are two important functions: `multiplicative_variance_trees`, `predict`

Here is an example:

```
from varvar.trees import multiplicative_variance_trees, predict  # takes time because numba compiles functions
import numpy as np

random = np.random.RandomState(1729)
n = 200000
x = random.uniform(-1000, 1000, n)
correct_threshold = 300
sigma = 1 * (x <= correct_threshold) + 30 * (x > correct_threshold)
e = sigma * random.randn(n)

trees = multiplicative_variance_trees(
    [x], e**2,
    num_trees=1, max_depth=1, mingain=1, learning_rate=1,
    q=np.linspace(0, 1, 100)[1:-1]
)
preds = predict(trees, [x])

found_threshold = trees[1][0][1]
print(correct_threshold, found_threshold)  # 300, 295
print(np.sqrt(min(preds)), np.sqrt(max(preds)))  # 1, 30
```
