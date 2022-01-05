import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.datasets import make_classification
import lightgbm as lgb
X, y = make_classification(100, n_classes=2, n_informative=3, n_redundant=0, n_clusters_per_class=2, n_features=10)
lgb_model = lgb.LGBMClassifier(n_estimators=3)
lgb_model.fit(X, y)
nclass = lgb_model.n_classes_
print(lgb_model.predict(X))



from treegrad.tree_utils import multi_tree_to_param

model_dump = lgb_model.booster_.dump_model()
trees_ = [m["tree_structure"] for m in model_dump["tree_info"]]
trees_params = multi_tree_to_param(X, y, trees_)

weights = trees_params[0]
sparse_info = trees_params[1]
routes_list = trees_params[2]

# https://arxiv.org/pdf/1909.09223.pdf