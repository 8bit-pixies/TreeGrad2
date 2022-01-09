"""
Uses a base EBM model for learning an architecture which we can then apply
transfer learning or fine-tuning

Reference: https://tensorflow.google.cn/recommenders/examples/featurization?hl=zh-cn
"""
from collections.abc import Iterable
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from interpret.glassbox import (
    ExplainableBoostingRegressor,
    ExplainableBoostingClassifier,
)


X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

ebm = ExplainableBoostingRegressor(random_state=seed)
ebm.fit(X_train, y_train)


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.bias = self.add_weight("bias", shape=(1,))

    def call(self, inputs):
        return tf.math.reduce_sum(tf.concat(inputs, axis=1), axis=1) + self.bias


class InteractionLayer(tf.keras.layers.Layer):
    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def build(self, input_shape):
        self.lookup = self.add_weight("interaction_lookup", shape=self.shape)

    def call(self, inputs):
        indx = tf.stack(inputs, axis=-1)
        return tf.gather_nd(self.lookup, indx)


class EBMModel(tf.keras.Model):
    def __init__(self, model=None, set_weights=False):
        super().__init__()
        self.model = model
        self.bias = BiasLayer()
        self.create_model()

    def create_model(self, set_weights=False):
        """
        See reference: https://github.com/interpretml/ebm2onnx/blob/master/ebm2onnx/convert.py
        """
        self.feature_model = []
        self.feature_info = []
        self.feature_names = []
        for feature_index in range(len(self.model.feature_names)):
            feature_name = self.model.feature_names[feature_index]
            feature_name = feature_name.replace(" ", "__")
            feature_type = self.model.feature_types[feature_index]
            feature_group = self.model.feature_groups_[feature_index]
            self.feature_names.append(feature_name)
            info_config = {}
            # print(feature_type)
            if feature_type == "continuous":
                self.feature_model.append(
                    tf.keras.Sequential(
                        [
                            tf.keras.layers.Discretization(
                                list(self.model.preprocessor_.col_bin_edges_[feature_group[0]])
                            ),
                            tf.keras.layers.IntegerLookup(
                                len(self.model.preprocessor_.col_bin_edges_[feature_group[0]]) + 1,
                                vocabulary=tf.constant(
                                    list(range(len(self.model.preprocessor_.col_bin_edges_[feature_group[0]])))
                                ),
                                output_mode="one_hot",
                                pad_to_max_tokens=True,
                            ),
                            tf.keras.layers.Dense(1, use_bias=False, name=f"{feature_name}"),
                        ]
                    )
                )
                info_config["feature_type"] = feature_type
                info_config["column_name"] = feature_name
                info_config["column_index"] = feature_group[0]
                info_config["scores"] = self.model.additive_terms_[feature_index][1:]
                self.feature_info.append(info_config)
            elif feature_type == "interaction":
                interactions = [self.model.feature_types[idx] for idx in feature_group]
                # else not implemented right now
                assert interactions[0] == "continuous"
                assert interactions[1] == "continuous"
                left_size = len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist()) + 1
                right_size = len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist()) + 1
                left_input = tf.keras.layers.Input(shape=(1,))
                left_x = tf.keras.layers.Discretization(
                    self.model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist()
                )(left_input)
                left_x = tf.keras.layers.IntegerLookup(
                    len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist()) + 1,
                    vocabulary=tf.constant(
                        list(range(len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist())))
                    ),
                )(left_x)
                right_input = tf.keras.layers.Input(shape=(1,))
                right_x = tf.keras.layers.Discretization(
                    self.model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist()
                )(right_input)
                right_x = tf.keras.layers.IntegerLookup(
                    len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist()) + 1,
                    vocabulary=tf.constant(
                        list(range(len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist())))
                    ),
                )(right_x)
                # left_input = tf.keras.Sequential([
                #     tf.keras.layers.Discretization(self.model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist()),
                #     tf.keras.layers.IntegerLookup(len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist()) + 1,
                #     vocabulary = tf.constant(list(range(len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist())))))
                # ])
                # right_input = tf.keras.Sequential([
                #     tf.keras.layers.Discretization(self.model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist()),
                #     tf.keras.layers.IntegerLookup(len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist()) + 1,
                #     vocabulary = tf.constant(list(range(len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist())))))
                # ])
                # print(left_input, right_input)
                # print(dir(left_input))
                output = InteractionLayer([left_size, right_size], name=f"{feature_name}")([left_x, right_x])
                self.feature_model.append(tf.keras.Model(inputs=[left_input, right_input], outputs=output))
                info_config["feature_type"] = feature_type
                info_config["column_name"] = [self.model.preprocessor_.feature_names[idx] for idx in feature_group]
                info_config["column_index"] = list(feature_group)
                # info_config["scores"] = self.model.additive_terms_[feature_index][1:]
                info_config["scores"] = self.model.explain_global().data(feature_index)["scores"]
                self.feature_info.append(info_config)

            else:
                # raise NotImplementedError("")
                continue

        if set_weights:
            for feature_index in range(len(self.feature_names)):
                nm = self.feature_names[feature_index]
                self.feature_model[feature_index].get_layer(nm).set_weights(self.feature_info[feature_index]["scores"])
            self.bias.set_weights(model.intercept_)

    def call(self, inputs):
        # print(zip(self.feature_model, self.feature_info))
        outputs = []

        if type(inputs) is pd.DataFrame:
            for mod, info in zip(self.feature_model, self.feature_info):
                cols = info["column_name"]
                if type(cols) is str:
                    outputs.append(mod(np.array(inputs[cols])))
                else:
                    outputs.append(mod([np.array(inputs[cols[0]]), np.array(inputs[cols[1]])]))
        else:
            for mod, info in zip(self.feature_model, self.feature_info):
                cols = info["column_index"]
                if type(cols) is str or type(cols) is int:
                    outputs.append(mod(inputs[:, cols]))
                else:
                    outputs.append(mod([inputs[:, cols[0]], inputs[:, cols[1]]]))

        return self.bias(outputs)


model = EBMModel(ebm)
# output = model(X_train)
output = model(X_train.values)

X_train[model.feature_info[-1]["column_name"]]
model.feature_model[-1](
    [
        X_train[model.feature_info[-1]["column_name"][0]],
        X_train[model.feature_info[-1]["column_name"][1]],
    ]
)

model.compile(loss="mse", optimizer="sgd")
model.fit(X_train.values, np.array(y_train), epochs=1)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_train, ebm.predict(X_train)))
print(mean_squared_error(np.array(y_train), model.predict(X_train.values)))

# overfits!
print(mean_squared_error(y_test, ebm.predict(X_test)))
print(mean_squared_error(np.array(y_test), model.predict(X_test.values)))

for feature_index in range(len(model.feature_names)):
    nm = model.feature_names[feature_index]
    print(model.feature_model[feature_index].get_layer(nm))


model2 = EBMModel(ebm, True)
print(mean_squared_error(np.array(y_train), model2.predict(X_train.values)))
print(mean_squared_error(np.array(y_test), model2.predict(X_test.values)))

# ebm.predict_proba(X_train)
model2.predict(X_train_dict)
