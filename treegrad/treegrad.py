import sys

import tensorflow as tf

interpret = None
lgb = None

if "interpret" in sys.modules:
    import interpret

    from treegrad.utils_interpret import EBMClassifier
if "lightgbm" in sys.modules:
    import lightgbm as lgb

    from treegrad.utils_lgb import LGBClassifier


def make_treegrad(model, **kwargs):
    # refactor this
    if model is None:
        raise ValueError("Model not provided!")
    elif interpret is not None:
        if type(model) is interpret.glassbox.ExplainableBoostingClassifier:
            return EBMClassifier(model, **kwargs)
    elif lgb is not None:
        if type(model) is lgb.LGBMClassifier:
            return LGBClassifier(model, **kwargs)


class TreeGradClassifier(tf.keras.Model):
    def __init__(self, model=None, **kwargs):
        super().__init__()
        self.base_model = make_treegrad(model, **kwargs)

    def call(self, inputs):
        return self.base_model.call(inputs)
