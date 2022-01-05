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