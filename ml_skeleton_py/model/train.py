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
from ml_skeleton_py import settings

logger = logging.getLogger(__name__)


def fetch_model(model: str) -> Tuple[BaseEstimator, dict]:
    """
    Fetch a model and the corresponding optimized hyperparameters.

    Parameters:
        model (str): lr  (logistic regression)
                     knn (k nearest neighbors)
                     svc (support vector machines)
                     dt  (decision tree classifier)

    Returns:
        classifier (BaseEstimator): classifier model
    """
    if model == "lr":
        classifier = LogisticRegression(max_iter=4000, penalty="l2", C=0.01)
    elif model == "knn":
        classifier = KNeighborsClassifier(n_neighbors=4, algorithm="auto")
    elif model == "svc":
        classifier = SVC(C=1, kernel="linear")
    elif model == "dt":
        classifier = DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_leaf=5)
    return classifier


def save_transformed_data(object: pd.DataFrame, file: str) -> None:
    """
    Saves a transformed data object.

    Parameters:
        object (pd.DataFrame): a dataframe that you want to save

        file (str): the filename of the object

    Return:
        None
    """
    with open(os.path.join(settings.DATA_TRANSFORMED, file), "wb") as handle:
        pickle.dump(object, handle)
    return None


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

    save_transformed_data(X_train, "X_train.p")
    save_transformed_data(y_train, "y_train.p")
    save_transformed_data(X_test, "X_test.p")
    save_transformed_data(y_test, "y_test.p")
    return X_train, y_train, X_test, y_test


def train(model: str, dataset: str) -> None:
    """
    Train models using X_train and y_train with a specific classifier.

    Trains a specific classifier with a set of optimized hyperparameters in a 5fold-CV.
    The training results with the accompanying model is saved in ./models/

    Parameters:
        model (str): the model that you want to train
                     options:
                        "lr": logistic regression
                        "knn": k nearest neighbors
                        "svc": support vector classifier
                        "dt": decision tree

        dataset (str): the dataset on which you want to train

    Returns:
        None

    """
    # loading data
    df = pd.read_csv(os.path.join(settings.DATA_RAW, dataset))
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
    # In this specific example logistic regression was chosen as the most optimal model
    # after running several experiments.
    classifier = fetch_model(model=model)

    # training
    classifier.fit(X_train, y_train)
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
    predict_pipeline = make_pipeline(scaler, classifier)
    pred_result = {
        "clf": model,
        "training score roc_auc": training_score.mean(),
        "model": predict_pipeline,
    }
    pickle.dump(pred_result, open(os.path.join(settings.MODEL_DIR, model) + ".p", "wb"))

