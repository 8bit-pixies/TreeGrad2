# TreeGrad2

`TreeGrad` implements a naive approach to converting a Gradient Boosted Tree Model to an Online trainable model. It does this by creating differentiable tree models which can be learned via auto-differentiable frameworks. This repository is an extension of the original TreeGrad algorithm and supports transfer learning of models created using Gradient Boosting Machines from `lightgbm` and Explainable Boosting Machines from `interpretml`.

# Usage

When using LightGBM

```py
import lightgbm as lgb
from treegrad import TreeGradClassifier

lgbm = lgb.LGBMClassifier()
lgbm.fit(X, y)


model = TreeGradClassifier(model=lgbm)
# if binary classification
model.compile(loss="binary_crossentropy", optimizer="sgd")
model.fit(X, y, batch_size=32, epochs=10)
```

When using EBM

```py
from interpret.glassbox import ExplainableBoostingClassifier
from treegrad import TreeGradClassifier

ebm = ExplainableBoostingClassifier()
ebm.fit(X, y)

model = TreeGradClassifier(model=ebm)
# if binary classification
model.compile(loss="binary_crossentropy", optimizer="sgd")
model.fit(X, y, batch_size=32, epochs=10)
```

# Performance

There are several engineering challenges to fine-tuning using the original treegrad when moving to tensorflow, hence the performance is not up to par. That being said the EBM implementation is much more straightforward and achieves fairly comparable results.


| dataset_name   |      ebm |   ebm_tf |      lgb |   lgb_tf |       lr |   rf-100 |
|:---------------|---------:|---------:|---------:|---------:|---------:|---------:|
| adult          | 0.866601 | 0.830119 | 0.869549 | 0.775458 | 0.846579 | 0.852352 |
| breast-cancer  | 0.937063 | 0.923077 | 0.965035 | 0.636364 | 0.993007 | 0.951049 |
| credit-fraud   | 0.999522 | 0.99934  | 0.996377 | 0.998329 | 0.999214 | 0.999579 |
| heart          | 0.894737 | 0.855263 | 0.789474 | 0.736842 | 0.868421 | 0.842105 |

The ability to perform stateful updates with minimal sacrifice to performance makes this approach worthwhile.