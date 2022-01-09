import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import warnings

import lightgbm as lgb
import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from treegrad.treegrad import make_treegrad
from sklearn.model_selection import train_test_split
import tqdm


from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import SGDClassifier, LogisticRegression

from interpret.glassbox import ExplainableBoostingClassifier

warnings.filterwarnings("ignore")
seed = 1337


def load_breast_data():
    breast = load_breast_cancer()
    feature_names = list(breast.feature_names)
    X, y = pd.DataFrame(breast.data, columns=feature_names), breast.target
    dataset = {
        "problem": "classification",
        "full": {
            "X": X,
            "y": y,
        },
    }
    return dataset


def load_adult_data():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None)
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
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]

    dataset = {
        "problem": "classification",
        "full": {
            "X": X_df,
            "y": y_df,
        },
    }

    return dataset


def load_heart_data():
    # https://www.kaggle.com/ronitf/heart-disease-uci
    df = pd.read_csv(r"data/heart.csv")
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]
    dataset = {
        "problem": "classification",
        "full": {
            "X": X_df,
            "y": y_df,
        },
    }

    return dataset


def load_credit_data():
    # https://www.kaggle.com/mlg-ulb/creditcardfraud
    df = pd.read_csv(r"data/creditcard.csv")
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]
    dataset = {
        "problem": "classification",
        "full": {
            "X": X_df,
            "y": y_df,
        },
    }

    return dataset


def load_telco_churn_data():
    # https://www.kaggle.com/blastchar/telco-customer-churn/downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv/1
    df = pd.read_csv(r"data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    train_cols = df.columns[1:-1]  # First column is an ID
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]  # 'Yes, No'
    dataset = {
        "problem": "classification",
        "full": {
            "X": X_df,
            "y": y_df,
        },
    }

    return dataset


def process_model(clf, name, X, y):
    # Evaluate model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1337)

    clf.fit(X_train, y_train)
    record = dict()
    record["model_name"] = name
    record["train_score"] = accuracy_score(y_train, clf.predict(X_train))
    record["test_score"] = accuracy_score(y_test, clf.predict(X_test))
    return record


def process_model_tf(name, X, y, preprocess):
    y = LabelEncoder().fit_transform(y)
    X = preprocess.fit_transform(X).astype(np.float32)

    callbacks = [tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=10)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1337)
    X_train = X_train + np.random.normal(loc=1e-16, size=X_train.shape)
    if name.startswith("ebm"):
        ebm = ExplainableBoostingClassifier(random_state=seed, feature_types=["continuous" for _ in range(X.shape[1])])
        ebm.fit(X_train, y_train)
        clf = make_treegrad(ebm, set_weights=True)
        clf.compile(loss="binary_crossentropy", optimizer="sgd")
        clf(X_train)
        callbacks = [tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=10)]
        clf.fit(X_train, y_train, verbose=0, epochs=100, validation_split=0.3, callbacks=callbacks)
    else:
        # lgb
        lgbm = lgb.LGBMClassifier(random_state=seed, n_estimators=10)
        lgbm.fit(X_train, y_train)
        clf = make_treegrad(lgbm, X=X_train, y=y_train, set_weights=True)
        clf.compile(loss="binary_crossentropy", optimizer="sgd")
        callbacks = [tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=10)]
        clf.fit(X_train, y_train, verbose=0, epochs=100, validation_split=0.3, callbacks=callbacks)
    record = dict()
    record["model_name"] = name
    record["train_score"] = accuracy_score(y_train, np.round(clf.predict(X_train)))
    record["test_score"] = accuracy_score(y_test, np.round(clf.predict(X_test)))
    return record


def benchmark_models(dataset_name, X, y, ct=None, n_splits=3, random_state=1337):
    if ct is None:
        is_cat = np.array([dt.kind == "O" for dt in X.dtypes])
        cat_cols = X.columns.values[is_cat]
        num_cols = X.columns.values[~is_cat]

        cat_ohe_step = ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"))

        cat_pipe = Pipeline([cat_ohe_step])
        num_pipe = Pipeline([("identity", FunctionTransformer())])
        transformers = [("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)]
        ct = ColumnTransformer(transformers=transformers)

    records = []

    summary_record = {}
    summary_record["dataset_name"] = dataset_name
    print()
    print("-" * 78)
    print(dataset_name)
    print("-" * 78)
    print(summary_record)
    print()

    pipe = Pipeline(
        [
            ("ct", ct),
            ("std", StandardScaler()),
            ("lr", LogisticRegression(random_state=random_state)),
        ]
    )
    record = process_model(pipe, "lr", X, y)
    print(record)
    record.update(summary_record)
    records.append(record)

    pipe = Pipeline(
        [
            ("ct", ct),
            # n_estimators updated from 10 to 100 due to sci-kit defaults changing in future versions
            ("rf-100", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)),
        ]
    )
    record = process_model(pipe, "rf-100", X, y)
    print(record)
    record.update(summary_record)
    records.append(record)

    pipe = Pipeline(
        [
            ("ct", ct),
            ("lgb", lgb.LGBMClassifier(random_state=random_state)),
        ]
    )
    record = process_model(pipe, "lgb", X, y)
    print(record)
    record.update(summary_record)
    records.append(record)

    # No pipeline needed due to EBM handling string datatypes
    ebm_inter = ExplainableBoostingClassifier(n_jobs=-1, random_state=random_state)
    record = process_model(ebm_inter, "ebm", X, y)
    print(record)
    record.update(summary_record)
    records.append(record)

    record = process_model_tf("ebm_tf", X, y, preprocess=make_pipeline(ct, VarianceThreshold()))
    print(record)
    record.update(summary_record)
    records.append(record)

    record = process_model_tf("lgb_tf", X, y, preprocess=make_pipeline(ct, VarianceThreshold()))
    print(record)
    record.update(summary_record)
    records.append(record)

    return records


n_splits = 3
results = []
dataset = load_heart_data()
result = benchmark_models("heart", dataset["full"]["X"], dataset["full"]["y"])
results.append(result)

dataset = load_breast_data()
result = benchmark_models("breast-cancer", dataset["full"]["X"], dataset["full"]["y"])
results.append(result)


dataset = load_adult_data()
result = benchmark_models("adult", dataset["full"]["X"], dataset["full"]["y"])
results.append(result)


dataset = load_credit_data()
result = benchmark_models("credit-fraud", dataset["full"]["X"], dataset["full"]["y"])
results.append(result)


# dataset = load_telco_churn_data()
# result = benchmark_models('telco-churn', dataset['full']['X'], dataset['full']['y'], n_splits=3)
# results.append(result)


records = [item for result in results for item in result]
record_df = pd.DataFrame.from_records(records)
record_df.to_csv("treegrad-perf-classification.csv")
print(record_df.pivot("dataset_name", "model_name", "test_score").to_markdown())
