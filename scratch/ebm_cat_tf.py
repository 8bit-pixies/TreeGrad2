"""
Uses a base EBM model for learning an architecture which we can then apply
transfer learning or fine-tuning

Reference: https://tensorflow.google.cn/recommenders/examples/featurization?hl=zh-cn

Adding categorical support...
"""
import copy
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

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None,
)
df.columns = [
    "Age",
    "WorkClass",
    "fnlwgt",
    "Education",
    "EducationNum",
    "MaritalStatus",
    "Occupation",
    "Relationship",
    "Race",
    "Gender",
    "CapitalGain",
    "CapitalLoss",
    "HoursPerWeek",
    "NativeCountry",
    "Income",
]
df = df.sample(frac=0.05)
train_cols = df.columns[0:-1]
label = df.columns[-1]
X = df[train_cols].copy()
y = df[label].copy()

# convert things so we only have float32
for d, c in zip(list(X.dtypes), X.columns):
    if d.type is not np.object_:
        X[c] = X[c].astype("float32")


seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

ebm = ExplainableBoostingClassifier(random_state=seed)
ebm.fit(X_train, y_train)


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, intercept=None, **kwargs):
        super().__init__(**kwargs)
        self.intercept = None
        self.bias = None
        self.build()

    def build(self, input_shape=None):
        if self.intercept is not None and self.bias is None:
            self.bias = self.add_weight(
                "bias",
                shape=(1,),
                trainable=True,
                initializer=tf.constant_initializer(self.intercept),
            )
        elif self.bias is None:
            self.bias = self.add_weight(
                "bias",
                shape=(1,),
                trainable=True,
            )

    def call(self, inputs):
        return tf.math.reduce_sum(tf.concat(inputs, axis=1), axis=1) + self.bias


class InteractionLayer(tf.keras.layers.Layer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def build(self, input_shape=None):
        self.lookup = self.add_weight("interaction_lookup", shape=self.shape)

    def call(self, inputs):
        indx = tf.stack(inputs, axis=-1)
        return tf.gather_nd(self.lookup, indx)


class EBMModel(tf.keras.Model):
    def __init__(self, model=None, set_weights=False):
        super().__init__()
        self.model = model
        if set_weights:
            self.bias = BiasLayer(name="bias", intercept=model.intercept_)
        else:
            self.bias = BiasLayer(name="bias")
        self.sigmoid = tf.keras.layers.Activation("sigmoid")
        self.create_model(set_weights)

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
                                list(self.model.preprocessor_.col_bin_edges_[feature_group[0]]),
                                input_shape=(1,),
                            ),
                            tf.keras.layers.IntegerLookup(
                                len(self.model.preprocessor_.col_bin_edges_[feature_group[0]]) + 1,
                                vocabulary=tf.constant(
                                    list(range(len(self.model.preprocessor_.col_bin_edges_[feature_group[0]])))
                                ),
                                output_mode="one_hot",
                                pad_to_max_tokens=True,
                            ),
                            tf.keras.layers.Dense(1, use_bias=False),
                        ],
                        name=f"model_{feature_name}",
                    )
                )
                info_config["feature_type"] = feature_type
                info_config["column_name"] = feature_name
                info_config["column_index"] = feature_group[0]
                info_config["scores"] = self.model.additive_terms_[feature_index][1:]
                self.feature_info.append(info_config)
            if feature_type == "categorical":
                self.feature_model.append(
                    tf.keras.Sequential(
                        [
                            tf.keras.layers.StringLookup(
                                max_tokens=len(list(self.model.preprocessor_.col_mapping_[feature_group[0]].keys()))
                                + 1,
                                vocabulary=list(self.model.preprocessor_.col_mapping_[feature_group[0]].keys()),
                                output_mode="one_hot",
                                pad_to_max_tokens=True,
                                input_shape=(1,),
                            ),
                            tf.keras.layers.Dense(1, use_bias=False),
                        ],
                        name=f"model_{feature_name}",
                    )
                )
                info_config["feature_type"] = feature_type
                info_config["column_name"] = feature_name
                info_config["column_index"] = feature_group[0]
                info_config["scores"] = self.model.additive_terms_[feature_index]
                self.feature_info.append(info_config)
            elif feature_type == "interaction":
                interactions = [self.model.feature_types[idx] for idx in feature_group]
                # else not implemented right now
                if interactions[0] == "continuous":
                    left_input = tf.keras.layers.Input(shape=(1,))
                    left_size = len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist()) + 1
                    left_slice = slice(0, left_size)
                    left_x = tf.keras.layers.Discretization(
                        self.model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist()
                    )(left_input)
                    left_x = tf.keras.layers.IntegerLookup(
                        len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist()) + 1,
                        vocabulary=tf.constant(
                            list(range(len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist())))
                        ),
                    )(left_x)
                elif interactions[0] == "categorical":
                    left_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
                    left_size = len(list(self.model.preprocessor_.col_mapping_[feature_group[0]].keys())) + 1
                    left_slice = slice(0, left_size)
                    left_x = tf.keras.layers.StringLookup(
                        max_tokens=len(list(self.model.preprocessor_.col_mapping_[feature_group[0]].keys())) + 1,
                        vocabulary=list(self.model.preprocessor_.col_mapping_[feature_group[0]].keys()),
                        pad_to_max_tokens=True,
                    )(left_input)
                    left_x = tf.keras.layers.IntegerLookup(
                        len(self.model.pair_preprocessor_.col_mapping_[feature_group[0]].keys()) + 1,
                        vocabulary=tf.constant(
                            list(range(len(self.model.pair_preprocessor_.col_mapping_[feature_group[0]].keys())))
                        ),
                    )(left_x)
                else:
                    raise ValueError("")

                if interactions[1] == "continuous":
                    right_input = tf.keras.layers.Input(shape=(1,))
                    right_size = len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist()) + 1
                    right_slice = slice(0, right_size)
                    right_x = tf.keras.layers.Discretization(
                        self.model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist()
                    )(right_input)
                    right_x = tf.keras.layers.IntegerLookup(
                        len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist()) + 1,
                        vocabulary=tf.constant(
                            list(range(len(self.model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist())))
                        ),
                    )(right_x)
                elif interactions[1] == "categorical":
                    right_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
                    right_size = len(list(self.model.preprocessor_.col_mapping_[feature_group[1]].keys())) + 1
                    right_slice = slice(0, right_size)
                    right_x = tf.keras.layers.StringLookup(
                        max_tokens=len(list(self.model.preprocessor_.col_mapping_[feature_group[1]].keys())) + 1,
                        vocabulary=list(self.model.preprocessor_.col_mapping_[feature_group[1]].keys()),
                        pad_to_max_tokens=True,
                    )(right_input)
                    right_x = tf.keras.layers.IntegerLookup(
                        len(self.model.pair_preprocessor_.col_mapping_[feature_group[1]].keys()) + 1,
                        vocabulary=tf.constant(
                            list(range(len(self.model.pair_preprocessor_.col_mapping_[feature_group[1]].keys())))
                        ),
                    )(right_x)
                else:
                    raise ValueError("")

                # assert interactions[0] == "continuous"
                # assert interactions[1] == "continuous"
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
                output = InteractionLayer([left_size, right_size])([left_x, right_x])
                self.feature_model.append(
                    tf.keras.Model(
                        inputs=[left_input, right_input],
                        outputs=output,
                        name=f"model_{feature_name}",
                    )
                )
                info_config["feature_type"] = interactions
                info_config["column_name"] = [self.model.preprocessor_.feature_names[idx] for idx in feature_group]
                info_config["column_index"] = list(feature_group)

                info_config["scores"] = self.model.additive_terms_[feature_index]  # [left_slice, right_slice]
                self.feature_info.append(info_config)
            else:
                # raise NotImplementedError("")
                continue

        if set_weights:
            for feature_index in range(len(self.feature_names)):
                nm = self.feature_names[feature_index]
                w = self.feature_model[feature_index].weights[0]
                learned_w = self.feature_info[feature_index]["scores"]
                try:
                    if len(self.feature_info[feature_index]["scores"].shape) == 2:
                        if w.shape[0] < learned_w.shape[0]:
                            learned_w = learned_w[1:, :]
                        if w.shape[1] < learned_w.shape[1]:
                            learned_w = learned_w[:, 1:]
                    else:
                        learned_w = learned_w.reshape(w.shape)
                    self.feature_model[feature_index].set_weights([learned_w])
                except:
                    print(nm, w.shape, learned_w.shape)
            self.bias.set_weights([np.array(self.model.intercept_).reshape((1,))])

    def call(self, inputs):
        # print(zip(self.feature_model, self.feature_info))
        outputs = []

        if type(inputs) in [pd.DataFrame]:
            for mod, info in zip(self.feature_model, self.feature_info):
                cols = info["column_name"]
                if type(cols) is str:
                    outputs.append(mod(tf.convert_to_tensor(inputs[cols].tolist)))
                else:
                    outputs.append(
                        mod(
                            [
                                tf.convert_to_tensor(inputs[cols[0]].tolist),
                                tf.convert_to_tensor(inputs[cols[1]].tolist),
                            ]
                        )
                    )
        elif type(inputs) in [dict]:
            for mod, info in zip(self.feature_model, self.feature_info):
                cols = info["column_name"]
                if type(cols) is str:
                    outputs.append(mod(inputs[cols]))
                else:
                    outputs.append(mod([inputs[cols[0]], inputs[cols[1]]]))
        else:
            for mod, info in zip(self.feature_model, self.feature_info):
                cols = info["column_index"]
                if type(cols) is str or type(cols) is int:
                    if info["feature_type"] == "continuous":
                        c_input = tf.cast(inputs[:, cols], tf.float32)
                    else:
                        c_input = tf.convert_to_tensor(inputs[:, cols])
                    outputs.append(mod(c_input))
                else:
                    if info["feature_type"][0] == "continuous":
                        l_input = tf.cast(inputs[:, cols[0]], tf.float32)
                    else:
                        l_input = tf.convert_to_tensor(inputs[:, cols[0]])

                    if info["feature_type"][1] == "continuous":
                        r_input = tf.cast(inputs[:, cols[1]], tf.float32)
                    else:
                        r_input = tf.convert_to_tensor(inputs[:, cols[1]])
                    outputs.append(mod([l_input, r_input]))
        # else:
        #     for mod, info in zip(self.feature_model, self.feature_info):
        #         cols = info["column_index"]
        #         if type(cols) is str or type(cols) is int:
        #             if info["feature_type"] == "continuous":
        #                 c_input = tf.cast(inputs[:, cols].tolist(), tf.float32)
        #             else:
        #                 c_input = tf.convert_to_tensor(inputs[:, cols].tolist())
        #             outputs.append(mod(c_input))
        #         else:
        #             if info["feature_type"][0] == "continuous":
        #                 l_input = tf.cast(inputs[:, cols[0]].tolist(), tf.float32)
        #             else:
        #                 l_input = tf.convert_to_tensor(inputs[:, cols[0]].tolist())

        #             if info["feature_type"][1] == "continuous":
        #                 r_input = tf.cast(inputs[:, cols[1]].tolist(), tf.float32)
        #             else:
        #                 r_input = tf.convert_to_tensor(inputs[:, cols[1]].tolist())
        #             outputs.append(mod([l_input, r_input]))

        pre_activation = self.bias(outputs)
        return self.sigmoid(pre_activation)


model = EBMModel(ebm)
# output = model(X_train)

# output = model(X_train.values)

X_train_dict = {}
for k in X_train.columns:
    X_train_dict[k] = tf.convert_to_tensor(X_train[k].tolist())
y_train_num = np.array(y_train == " >50K").astype(np.float32)
model(X_train_dict)

model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(
    X_train_dict,
    np.array(y_train == " >50K").astype(np.float32),
    verbose=1,
    batch_size=5,
    epochs=1,
)

from sklearn.metrics import accuracy_score, f1_score

model2 = EBMModel(ebm, True)
print(
    "base model",
    accuracy_score(y_train_num, np.round(model.predict(X_train_dict))),
    f1_score(y_train_num, np.round(model.predict(X_train_dict))),
)
print(
    "transfer model",
    accuracy_score(y_train_num, np.round(model2.predict(X_train_dict))),
    f1_score(y_train_num, np.round(model2.predict(X_train_dict))),
)

print(
    "ebm model",
    accuracy_score(y_train_num, ebm.predict(X_train) == " >50K"),
    f1_score(y_train_num, ebm.predict(X_train) == " >50K"),
)

ebm.predict_proba(X_train)
model2.predict(X_train_dict)

model2.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model2.fit(
    X_train_dict,
    np.array(y_train == " >50K").astype(np.float32),
    verbose=1,
    batch_size=32,
    epochs=1000,
)
print(
    "transfer model",
    accuracy_score(y_train_num, np.round(model2.predict(X_train_dict))),
    f1_score(y_train_num, np.round(model2.predict(X_train_dict))),
)

X_test_dict = {}
for k in X_train.columns:
    X_test_dict[k] = tf.convert_to_tensor(X_test[k].tolist())
y_test_num = np.array(y_test == " >50K").astype(np.float32)


print(
    "transfer model",
    accuracy_score(y_test_num, np.round(model2.predict(X_test_dict))),
    f1_score(y_test_num, np.round(model2.predict(X_test_dict))),
)

print(
    "ebm model",
    accuracy_score(y_test_num, ebm.predict(X_test) == " >50K"),
    f1_score(y_test_num, ebm.predict(X_test) == " >50K"),
)
