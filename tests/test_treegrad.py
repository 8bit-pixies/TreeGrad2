import lightgbm as lgb
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

from treegrad import __version__
from treegrad.treegrad import TreeGradClassifier, make_treegrad
from treegrad.utils_lgb import LGBClassifier


def test_version():
    assert __version__ == "0.1.0"


def test_treegrad_interpret_func():
    seed = 1
    X, y = datasets.make_moons(random_state=seed)
    ebm = ExplainableBoostingClassifier(random_state=seed)
    ebm.fit(X, y)
    model = make_treegrad(ebm, set_weights=True)
    assert np.abs(accuracy_score(y, ebm.predict(X)) - accuracy_score(y, np.round(model.predict(X)))) < 0.2


def test_treegrad_interpret_class():
    seed = 1
    X, y = datasets.make_moons(random_state=seed)
    ebm = ExplainableBoostingClassifier(random_state=seed)
    ebm.fit(X, y)
    model = TreeGradClassifier(ebm, set_weights=True)
    assert np.abs(accuracy_score(y, ebm.predict(X)) - accuracy_score(y, np.round(model.predict(X)))) < 0.2


def test_treegrad_tf_func():
    seed = 1
    X, y = datasets.make_moons(random_state=seed)
    lgb_model = lgb.LGBMClassifier(n_estimators=3)
    lgb_model.fit(X, y)

    # clearly the interface is broken here...
    model = LGBClassifier(lgb_model, X=X, y=y, set_weights=True)
    assert accuracy_score(y, np.round(model.predict(X))) >= 0.5


def test_treegrad_tf_func2():
    seed = 1
    X, y = datasets.make_classification(
        100, n_classes=2, n_informative=3, n_redundant=0, n_clusters_per_class=2, n_features=10, random_state=seed
    )
    lgb_model = lgb.LGBMClassifier(n_estimators=3)
    lgb_model.fit(X, y)

    # clearly the interface is broken here...
    model = LGBClassifier(lgb_model, X=X, y=y, set_weights=True)
    assert accuracy_score(y, np.round(model.predict(X))) >= 0.5
