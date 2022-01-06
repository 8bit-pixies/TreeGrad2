import copy
import itertools
from functools import reduce

import numpy
import numpy as np
import scipy.sparse
import tensorflow as tf
from scipy.special import expit


def get_route(tree):
    # gets the route for the tree...
    tree_dict = {}
    boundary_dict = {}
    leaf_dict = {}

    def recurse(sub_tree, child_split=None, parent=None):
        # pprint.pprint(sub_tree)
        route_path = {}

        if "threshold" in sub_tree:
            boundary_dict[sub_tree["split_index"]] = {
                "column": sub_tree["split_feature"],
                "value": sub_tree["threshold"],
            }

            if "split_index" in sub_tree["left_child"]:
                boundary_dict[sub_tree["split_index"]]["left"] = sub_tree["left_child"]["split_index"]

            if "split_index" in sub_tree["right_child"]:
                boundary_dict[sub_tree["split_index"]]["right"] = sub_tree["right_child"]["split_index"]
        else:
            # we're a leaf!
            leaf_dict[parent] = leaf_dict.get(parent, {})
            leaf_dict[parent][child_split] = sub_tree

        if "left_child" in sub_tree:
            try:
                route_path["left"] = sub_tree["left_child"]["split_index"]
            except Exception:
                # print("\tleft_child {}".format(e))
                pass
        if "right_child" in sub_tree:
            try:
                route_path["right"] = sub_tree["right_child"]["split_index"]
            except Exception:
                pass

        # print(route_path)
        if len(route_path) > 0:
            tree_dict[sub_tree["split_index"]] = route_path.copy()

        if "left_child" in sub_tree:
            recurse(sub_tree["left_child"], "left", sub_tree["split_index"])
        if "right_child" in sub_tree:
            recurse(sub_tree["right_child"], "right", sub_tree["split_index"])
        # print("\n\n")

    recurse(tree)

    # combine leaf_dict and boundary_dict
    max_index = np.max(list(boundary_dict.keys()))

    for k in leaf_dict.keys():
        # print(leaf_dict)
        if "left" in leaf_dict[k]:
            max_index += 1
            boundary_dict[k]["left"] = max_index
            tree_dict[k] = tree_dict.get(k, {})
            tree_dict[k]["left"] = max_index
            tree_dict[max_index] = {}
            pred_val = expit(leaf_dict[k]["left"]["leaf_value"])
            boundary_dict[max_index] = {
                "predict": np.array([1 - pred_val, pred_val]),
                "leaf_value": leaf_dict[k]["left"]["leaf_value"],
            }

        if "right" in leaf_dict[k]:
            max_index += 1
            boundary_dict[k]["right"] = max_index
            tree_dict[k] = tree_dict.get(k, {})
            tree_dict[k]["right"] = max_index
            tree_dict[max_index] = {}
            # print(leaf_dict)
            pred_val = expit(leaf_dict[k]["right"]["leaf_value"])
            boundary_dict[max_index] = {
                "predict": np.array([1 - pred_val, pred_val]),
                "leaf_value": leaf_dict[k]["right"]["leaf_value"],
            }

    return tree_dict, boundary_dict, leaf_dict


class BaseTree(object):
    def build_tree(self, depth=2):
        """
        builds the adjancey list up to depth of 2
        """
        total_nodes = np.sum([2 ** x for x in range(depth)])
        nodes = list(range(total_nodes))
        nodes_per_level = np.cumsum([2 ** x for x in range(depth - 1)])
        nodes_level = [x.tolist() for x in np.array_split(nodes, nodes_per_level)]

        adj_list = dict((idx, {}) for idx in nodes)
        for fr in nodes_level[:-1]:
            for i in fr:
                i_list = adj_list.get(i, {})
                # the connected nodes always follows this pattern
                i_list["left"] = i * 2 + 1
                i_list["right"] = i * 2 + 2
                adj_list[i] = i_list.copy()
        return adj_list

    def calculate_routes(self, adj_list=None):
        """
        Calculates routes in GBM format.

        {0:{'left': 1, 'right': 2}, 1:{}, 2:{}}                      --> [([(0, 0)], 1),
                                                                          ([(0, 1)], 2)]
        {0:{'left': 1, 'right': 2}, 1:{'left': 3, 'right':4},
         2:{}, 3:{}, 4: {}}                                          --> [([(0, 0), (1, 0)], 3),
                                                                          ([(0, 0), (1, 1)], 4),
                                                                          ([(0, 1)], 2)]
        """
        if adj_list is None:
            adj_list = self.build_tree(3)

        def get_next(next_node, current_path):
            paths = adj_list[next_node]
            if len(paths) == 0:
                all_paths.append((current_path, next_node))
            else:
                # do left...
                get_next(paths["left"], current_path + [(next_node, 0)])
                get_next(paths["right"], current_path + [(next_node, 1)])

        all_paths = []
        get_next(0, [])
        return all_paths


class Tree(BaseTree):
    """
    Tree object to help abstract out some of the methods that are commonly used.
    Also used to help figure out how to maintain state around pruning and grafting nodes

    Usage:
    tt = Tree().graft()
    tt.plot()
    """

    def __init__(self, depth=3, nodes=None, tree=None, previous_state={}):
        self.depth = depth
        self.nodes = nodes if nodes is not None else np.sum([2 ** x for x in range(self.depth)])
        self.tree = tree if tree is not None else self.build_tree(self.depth)
        self.update()

    def update(self):
        self.update_route()
        self.update_nodes()

    def update_nodes(self):
        self.nodes = len([k for k, v in self.tree.items() if len(v) > 0])

    def update_depth(self):
        all_routes = [len(r) for r, _ in self.route]
        self.depth = max(all_routes)

    def update_route(self):
        self.route = self.calculate_routes(self.tree)
        self.route.sort(key=lambda x: x[1])


def flatten(ll):
    return [item for sublist in ll for item in sublist]


def split_trees_by_classes(trees, n_classes):
    # https://github.com/BayesWitnesses/m2cgen/blob/master/m2cgen/assemblers/boosting.py
    # Splits are computed based on a comment
    # https://github.com/dmlc/xgboost/issues/1746#issuecomment-267400592.
    if n_classes == 2:
        return trees

    trees_by_classes = [[] for _ in range(n_classes)]
    for i in range(len(trees)):
        class_idx = i % n_classes
        trees_by_classes[class_idx].append(trees[i])
    return trees_by_classes


def calculate_boundary(X, boundary_value, column):
    """
    We probably want to take the range of X[column]
    to determine approximately the correct value
    """
    x_range = np.max(X[:, column]) - np.min(X[:, column])
    coef_ = 1
    if np.abs(boundary_value) > 0.1 and np.abs(x_range) > 1:
        coef_ = x_range / 2
        inter = [coef_ * -boundary_value]
    else:
        coef_ = max(x_range / 2, 1.0)
        inter = [coef_ * -boundary_value]

    # print(coef_, inter)
    coef = ([coef_], [0], [column])
    return coef, inter


def boundary_weights(X, y, tree_dict, boundary_dict, output="dict"):
    """
    This recursively goes down the nodes and returns the
    results by extending boundary dict(?)
    """
    tree_temp = Tree(tree=tree_dict)

    unseen_nodes = [key for key, val in tree_dict.items() if len(val) > 0]

    route_path = [x[0] for x in tree_temp.route]
    route_path = [[x[0] for x in x_path] for x_path in route_path]
    route_path.sort()
    route_path = list(k for k, _ in itertools.groupby(route_path))

    for node in unseen_nodes:
        coef, inter = calculate_boundary(X, boundary_dict[node]["value"], boundary_dict[node]["column"])
        boundary_dict[node]["coef"] = coef
        boundary_dict[node]["inter"] = inter

    if output == "dict":
        return boundary_dict

    weights = []
    inter = []
    pred = []
    for idx in sorted(list(boundary_dict.keys())):
        if "coef" in boundary_dict[idx]:
            weights.append(boundary_dict[idx]["coef"])
            inter.append(boundary_dict[idx]["inter"])
        else:
            pred.append(boundary_dict[idx]["predict"])

    # weights = np.hstack(weights)
    # inter = np.hstack(inter)
    # pred = np.vstack(pred)
    return [(weights), (inter), (pred)]


# from tree build the sparse coef representation
def tree_to_nnet(X, y, tree):
    """
    Outputs the parameters for neural network
    """
    t_d, b_d, _ = get_route(tree)
    tt = Tree(tree=t_d)
    boundary_dict = boundary_weights(X, y, t_d, b_d, output="dict")
    return boundary_dict, tt.route


def boundary_dict_mapping(X, boundary_dict, mode="raw"):
    num_cols = X.shape[1]
    weights = []
    inter = []
    pred = []

    coef_mapping = []
    for idx in sorted(list(boundary_dict.keys())):
        if "coef" in boundary_dict[idx]:
            data, row, col = boundary_dict[idx]["coef"]
            weights_ = scipy.sparse.coo_matrix((data, (row, col)), shape=[1, num_cols])
            weights.append(np.array(weights_.todense()))
            inter.append(np.array(boundary_dict[idx]["inter"]))
            coef_mapping.append(idx)
        else:
            if mode == "proba":
                pred.append(boundary_dict[idx]["predict"])
            elif mode == "raw":
                pred.append(boundary_dict[idx]["leaf_value"])
            else:
                raise Exception("Expecting mode in ['proba', 'raw'], got {}".format(mode))
    return [(weights), (inter), (pred)], coef_mapping


def sigmoid(z):
    z = np.clip(z, -32, 32)
    return 1.0 / (1 + np.exp(-z))


def proba_to_alpha(proba=0.1):
    return proba / (1 - proba)


def alpha_to_proba(alpha=0.9):
    return alpha / (alpha + 1)


def old_route_to_new_route(route, num_nodes):
    route_array = []
    try:
        for rout, _ in route:
            data = []
            row = []
            col = []
            for node, direction in rout:
                data.append(1)
                row.append(0)
                # sprint(node)
                col.append(node if direction == 0 else node + num_nodes)
            route_array.append(scipy.sparse.coo_matrix((data, (row, col)), shape=(1, num_nodes * 2)).toarray())
        return np.vstack(route_array)
    except Exception:
        return None


def tree_to_param(X, y, tree):
    boundary, route = tree_to_nnet(X, y, tree)
    boundary_test = boundary_dict_mapping(X, boundary)
    params, _ = boundary_test
    coef_, inter_, leaf = params

    coef = np.vstack(coef_).T
    sparse_coef = scipy.sparse.coo_matrix(coef)
    inter = np.hstack(inter_)
    param = (sparse_coef.data, inter, np.array(leaf))
    # return param, (coef != 0)*1.0, route
    return param, (coef != 0) * 1.0, old_route_to_new_route(route, param[0].shape[0])


def multi_tree_to_param(X, y, trees):
    param_route = [tree_to_param(X, y, tree) for tree in trees]
    all_param = flatten([x[0] for x in param_route])
    all_sparse_info = [x[1] for x in param_route]
    all_route = [x[2] for x in param_route]
    return all_param, all_sparse_info, all_route


def multiclass_trees_to_param(X, y, multitrees):
    param_list = [multi_tree_to_param(X, y, tree_x) for tree_x in multitrees]
    all_param = flatten([x[0] for x in param_list])
    all_sparse_info = [x[1] for x in param_list]
    all_route = [x[2] for x in param_list]
    return all_param, all_sparse_info, all_route


def gbm_gen(
    param=None,
    X=None,
    all_route=None,
    all_sparse_info=None,
    multi=False,
    num_classes=3,
    tau=0.01,
    eps=1e-11,
):
    def decision_tree(param, X, route, sparse_info, tau=tau, eps=eps):
        coef, inter, leaf = param
        # coef_sparse = scipy.sparse.coo_matrix((coef, (sparse_row_col[0], sparse_row_col[1])), shape=(X.shape[1], inter.shape[0])).todense() # this is just a hack

        # no sparsity
        coef_sparse = sparse_info * coef
        decisions = np.dot(X, np.hstack([coef_sparse, -coef_sparse])) + np.hstack([inter, -inter])
        decision_soft = np.log(gumbel_softmax(decisions, tau=tau) + eps)
        route_probas = np.exp(np.dot(decision_soft, route.T))
        proba = np.dot(route_probas, leaf)
        return proba

    # boosted_tree = reduce(lambda x, y: x+y, [decision_tree(X, y, tree) for tree in trees])
    def boosted_tree(all_param, X, all_route=all_route, all_sparse_info=all_sparse_info):
        # roll up params

        tree_pred = []
        num_unpack = 3
        for idx in range(0, len(all_param), num_unpack):
            param = all_param[idx : idx + num_unpack]
            # print(len(param))
            other_idx = idx // num_unpack
            # print(idx, other_idx)
            tree_pred.append(decision_tree(param, X, all_route[other_idx], all_sparse_info[other_idx]))
        pred_out = reduce(lambda x, y: x + y, tree_pred)
        return pred_out

    def multi_tree(
        all_param,
        X,
        all_route=all_route,
        all_sparse_info=all_sparse_info,
        num_classes=num_classes,
    ):
        # this is for multiclass trees
        # we need to unpack for each class...and then run the boosted tree
        num_unpack = len(all_param) // num_classes
        class_prediction = []
        for idx in range(0, len(all_param), num_unpack):
            other_idx = idx // num_unpack
            booster_param = all_param[idx : idx + num_unpack]
            booster_route = all_route[other_idx]
            booster_sparse_info = all_sparse_info[other_idx]
            class_prediction.append(boosted_tree(booster_param, X, booster_route, booster_sparse_info))
        class_prediction = [np.exp(x) for x in class_prediction]
        class_total = reduce(lambda x, y: x + y, class_prediction)

        return np.stack([x / class_total for x in class_prediction], axis=-1)

    if not multi:
        return boosted_tree
    else:
        return multi_tree


def simple_callback(params, t, g):
    if (t + 1) % 1 == 0:
        print("Iteration {}".format(t + 1))


def update_tree_info(tree, split_index, threshold, split_feature=None):
    def update_level(tree, split_index, threshold, split_feature=None):
        if tree.get("split_index", None) == split_index:
            tree["threshold"] = threshold
            if split_feature is not None:
                tree["split_feature"] = split_feature
            return tree.copy()
        return tree.copy()

    def traverse_dict(tree, split_index, threshold, split_feature):
        tree = update_level(tree.copy(), split_index, threshold, split_feature)
        for k, v in tree.items():
            if isinstance(v, dict):
                tree[k] = traverse_dict(v.copy(), split_index, threshold, split_feature)
        return tree

    tree_copy = tree.copy()
    return traverse_dict(tree_copy.copy(), split_index, threshold, split_feature)


def update_leaf_info(tree, leaf_index, leaf_value):
    def update_level(tree, leaf_index, leaf_value):
        if tree.get("leaf_index", None) == leaf_index:
            tree["leaf_value"] = leaf_value
            return tree.copy()
        return tree.copy()

    def traverse_dict(tree, leaf_index, leaf_value):
        tree = update_level(tree.copy(), leaf_index, leaf_value)
        for k, v in tree.items():
            if isinstance(v, dict):
                tree[k] = traverse_dict(v.copy(), leaf_index, leaf_value)
        return tree

    tree_copy = tree.copy()
    return traverse_dict(tree_copy.copy(), leaf_index, leaf_value)


def param_to_tree(param, sparse_info, single_tree):
    # we want to iterate over model info and replace it so that we have something that can be compared
    # this only works for binary class for now
    # this is due to laziness, and difficulty to compare multiclass?
    coef, inter, leaf = param
    thresholds = (-inter / coef).tolist()
    leaves = (leaf).tolist()
    tree = copy.deepcopy(single_tree)

    # now figure out the columns
    feat_splits = []
    for col_idx in range(sparse_info.shape[1]):
        split_indx = np.nonzero(sparse_info[:, col_idx])[0].tolist()[0]
        feat_splits.append(split_indx)

    # now combine it all based on model_info...
    for idx, threshold in enumerate(thresholds):
        # iterate through the tree and replace where
        # split_index == idx
        tree = update_tree_info(tree.copy(), idx, threshold, feat_splits[idx])

    for idx, value in enumerate(leaves):
        # iterate through the tree and replace where
        # split_index == idx
        tree = update_leaf_info(tree.copy(), idx, value)
    return tree


class NodeLayer(tf.keras.layers.Layer):
    def __init__(self, num_nodes, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes

    def build(self, input_shape):
        # we may want a sparse one later...ignore it for now
        self.kernel = self.add_weight(
            "sparse", shape=[int(input_shape[-1]), self.num_nodes], regularizer="l1", trainable=True
        )
        self.bias = self.add_weight(
            "bias",
            constraint=tf.keras.constraints.MinMaxNorm(1.0, 2.0),
            shape=[
                self.num_nodes,
            ],
        )

    def call(self, input):
        return tf.matmul(input, tf.concat([self.kernel, -self.kernel], 1)) + tf.concat([self.bias, -self.bias], 0)


def gumbel_softmax(x, tau=0.01):
    x_temp = tf.clip_by_value(x / tau, -32, 32)
    return 1 / (1 + tf.keras.backend.exp(-x_temp))


def activation1(x):
    return tf.keras.backend.log(tf.clip_by_value(gumbel_softmax(x, 0.01), tf.keras.backend.epsilon(), 1.0))


def activation2(x):
    return tf.keras.backend.exp(tf.clip_by_value(x, -32, 0))


class LGBClassifier(tf.keras.Model):
    def __init__(self, model=None, X=None, y=None, set_weights=False):
        super().__init__()
        self.base_model = model
        self.create_model(X, y, set_weights)

    def create_model(self, X=None, y=None, set_weights=False):
        # nclass = self.base_model.n_classes_
        model_dump = self.base_model.booster_.dump_model()
        nfeats = len(model_dump["feature_names"])
        trees_ = [m["tree_structure"] for m in model_dump["tree_info"]]
        trees_params = multi_tree_to_param(X, y, trees_)
        self.trees_params = trees_params
        inputs = tf.keras.Input(shape=(nfeats,))
        decision_tree_layers = []
        for tree_idx in range(len(trees_)):
            num_nodes = trees_params[1][tree_idx].shape[1]
            # num_nodes = sparse_info[tree_idx].shape[1]
            decision_tree_layers.append(
                tf.keras.Sequential(
                    [
                        NodeLayer(num_nodes, name=f"node_{tree_idx}"),
                        tf.keras.layers.Lambda(activation1),
                        tf.keras.layers.Dense(num_nodes + 1, trainable=False, use_bias=False, name=f"route_{tree_idx}"),
                        # tf.keras.layers.Lambda(lambda x: tf.matmul(x, routes_list[tree_idx].T.astype(np.float32))),
                        tf.keras.layers.Lambda(activation2),
                        tf.keras.layers.Dense(1, use_bias=False, name=f"leaf_{tree_idx}"),
                    ],
                    name=f"decision_tree_{tree_idx}",
                )(inputs)
            )
        if len(decision_tree_layers) > 1:
            log_preds = tf.keras.layers.Add()(decision_tree_layers)
        else:
            log_preds = decision_tree_layers[0]
        log_preds = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -32, 32))(log_preds)
        preds = tf.keras.layers.Activation("sigmoid")(log_preds)
        self.model = tf.keras.Model(inputs=inputs, outputs=preds)

        if set_weights:
            for tree_idx in range(len(trees_)):
                coef, inter, leaf = trees_params[0][tree_idx * 3 : (tree_idx + 1) * 3]
                sparse_info = trees_params[1][tree_idx]
                route_array = trees_params[2][tree_idx]
                coef_sparse = sparse_info * coef
                if len(leaf.shape) == 1:
                    leaf = leaf[:, np.newaxis]
                self.model.get_layer(f"decision_tree_{tree_idx}").get_layer(f"node_{tree_idx}").set_weights(
                    [coef_sparse, inter]
                )
                self.model.get_layer(f"decision_tree_{tree_idx}").get_layer(f"route_{tree_idx}").set_weights(
                    [route_array.T]
                )
                self.model.get_layer(f"decision_tree_{tree_idx}").get_layer(f"leaf_{tree_idx}").set_weights([-leaf])

    def call(self, inputs):
        return self.model(inputs)
