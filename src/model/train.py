"""
EXAMPLE of training procedure on a highly imbalanced credit fraud dataset.

The dataset is retrieved from:
https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets.
"""

from typing import Tuple
import os
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import logging
from config import settings as s

logger = logging.getLogger(__name__)


def fetch_model(model: str) -> Tuple[BaseEstimator, dict]:
    """
    Fetch a model and the corresponding grid parameters to be searched.

    Parameters:
        model (str): lr  (logistic regression)
                     knn (k nearest neighbors)
                     svc (support vector machines)
                     dt  (decision tree classifier)

    Returns:
        classifier (BaseEstimator): classifier model

        params (dict): hyperparameters for grid search.

    """
    if model == "lr":
        classifier = (LogisticRegression(max_iter=4000),)
        params = {"penalty": ["l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    elif model == "knn":
        classifier = KNeighborsClassifier()
        params = {
            "n_neighbors": list(range(2, 5, 1)),
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        }
    elif model == "svc":
        classifier = SVC()
        params = {
            "C": [0.5, 0.7, 0.9, 1],
            "kernel": ["rbf", "poly", "sigmoid", "linear"],
        }
    elif model == "dt":
        classifier = DecisionTreeClassifier()
        params = {
            "criterion": ["gini", "entropy"],
            "max_depth": list(range(2, 4, 1)),
            "min_samples_leaf": list(range(5, 7, 1)),
        }
    return classifier, params


def save_split_data(
    X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the datasets into training and test and saves it in the transformed data folder.

    Parameters:
        X: the dataset

        y: the labels

    Returns:
        X_train (pd.DataFrame): the training dataset

        y_train (pd.DataFrame): the training labels

        X_test (pd.DataFrame): the test dataset

        y_test (pd.DataFrame): the test labels
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pickle.dump(X_train, open(os.path.join(s.DATA_TRANSFORMED, "X_train.p"), "wb"))
    pickle.dump(y_train, open(os.path.join(s.DATA_TRANSFORMED, "y_train.p"), "wb"))
    pickle.dump(X_test, open(os.path.join(s.DATA_TRANSFORMED, "X_test.p"), "wb"))
    pickle.dump(y_test, open(os.path.join(s.DATA_TRANSFORMED, "y_test.p"), "wb"))
    return X_train, y_train, X_test, y_test


def train() -> None:
    """
    Train models using X_train and y_train with a specific classifier.

    Trains a specific classifier with a grid of parameters in a 5fold-CV.
    The training results with the accompanying model is saved in ./models/

    Returns:
        None

    """
    # loading data
    df = pd.read_csv(os.path.join(s.DATA_RAW, "creditcard.csv"))
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # preprocessing
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    rus = RandomUnderSampler(replacement=False)
    X, y = rus.fit_resample(X, y)

    # splitting in train and test
    X_train, y_train, X_test, y_test = save_split_data(X, y)

    # fetching model params
    model = "lr"
    classifier, params = fetch_model(model=model)
    classifier = classifier[0]

    # training
    grid_clf = GridSearchCV(classifier, params, scoring="roc_auc")
    grid_clf.fit(X_train, y_train)
    training_score = cross_val_score(
        classifier, X_train, y_train, cv=5, scoring="roc_auc"
    )
    print(
        "Classifier: ",
        classifier.__class__.__name__,
        "Has a training score of",
        round(training_score.mean(), 2) * 100,
        "% roc_auc",
    )

    # saving
    predict_pipeline = make_pipeline(scaler, grid_clf)
    pred_result = {
        "clf": model,
        "training score roc_auc": training_score.mean(),
        "model": predict_pipeline,
    }
    pickle.dump(pred_result, open(os.path.join(s.MODEL_DIR, model) + ".p", "wb"))


if __name__ == "__main__":
    train()
