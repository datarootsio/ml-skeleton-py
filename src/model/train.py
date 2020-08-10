# EXAMPLE adapted from
# https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

import pickle
from preprocess_wk import generate_dataset

import logging
from .. import settings as s

logger = logging.getLogger(__name__)


def load_dataset():
    X_train = pickle.load(open(os.path.join(s.DATA_TRANSFORMED, "X_train.p"), "rb"))
    y_train = pickle.load(open(os.path.join(s.DATA_TRANSFORMED, "y_train.p"), "rb"))
    return X_train, y_train


def train():
    # Let's implement simple classifiers

    X_train, y_train = load_dataset()

    classifiers = {
        "lr": LogisticRegression(max_iter=4000),
        "knn": KNeighborsClassifier(),
        "svc": SVC(),
        "dt": DecisionTreeClassifier(),
    }

    # Use GridSearchCV to find the best parameters.
    for key, classifier in classifiers.items():
        # classifier.fit(X_train, y_train)
        if key == "lr":
            params = {"penalty": ["l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        elif key == "knn":
            params = {
                "n_neighbors": list(range(2, 5, 1)),
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            }
        elif key == "svc":
            params = {
                "C": [0.5, 0.7, 0.9, 1],
                "kernel": ["rbf", "poly", "sigmoid", "linear"],
            }
        elif key == "dt":
            params = {
                "criterion": ["gini", "entropy"],
                "max_depth": list(range(2, 4, 1)),
                "min_samples_leaf": list(range(5, 7, 1)),
            }
        else:
            continue

        grid_clf = GridSearchCV(classifier, params)
        grid_clf.fit(X_train, y_train)
        best_classifier = grid_clf.best_estimator_
        training_score = cross_val_score(classifier, X_train, y_train, cv=5)
        print(
            "Classifier: ",
            classifier.__class__.__name__,
            "Has a training score of",
            round(training_score.mean(), 2) * 100,
            "% accuracy score",
        )

        pred_result = {
            "clf": key,
            "training score": training_score.mean(),
            "model": best_classifier,
        }
        pickle.dump(pred_result, open(os.path.join(s.MODEL_DIR, key) + ".p", "wb"))


if __name__ == "__main__":
    train()
