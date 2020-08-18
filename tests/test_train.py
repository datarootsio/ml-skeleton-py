import os
from sklearn.base import is_classifier, is_regressor
from ml_skeleton_py import settings as s
from ml_skeleton_py.model.train import fetch_model
from ml_skeleton_py.model.train import train
import pickle

DATASET = "creditcard.csv"


def test_lr():
    """
    Test logistic regression call for model.
    """
    model = "lr"
    classifier = fetch_model(model)
    assert is_classifier(classifier) or is_regressor(classifier)


def test_knn():
    """
    Test logistic regression call for model.
    """
    model = "knn"
    classifier = fetch_model(model)
    assert is_classifier(classifier) or is_regressor(classifier)


def test_svc():
    """
    Test logistic regression call for model.
    """
    model = "svc"
    classifier = fetch_model(model)
    assert is_classifier(classifier) or is_regressor(classifier)


def test_dt():
    """
    Test logistic regression call for model.
    """
    model = "dt"
    classifier = fetch_model(model)
    assert is_classifier(classifier) or is_regressor(classifier)


def test_train_lr():
    """
    Test whether logistic regression is trained and can be loaded.
    """
    model = "lr"
    train(model, DATASET)
    with open(os.path.join(s.MODEL_DIR, model) + ".p", "rb") as handle:
        pred_result = pickle.load(handle)
    classifier = is_classifier(pred_result["model"])
    regressor = is_regressor(pred_result["model"])
    assert classifier or regressor


def test_train_knn():
    """
    Test whether k nearest neighbors is trained and can be loaded.
    """
    model = "knn"
    train(model, DATASET)
    with open(os.path.join(s.MODEL_DIR, model) + ".p", "rb") as handle:
        pred_result = pickle.load(handle)
    classifier = is_classifier(pred_result["model"])
    regressor = is_regressor(pred_result["model"])
    assert classifier or regressor


def test_train_svc():
    """
    Test whether support vector classifier is trained and can be loaded.
    """
    model = "svc"
    train(model, DATASET)
    with open(os.path.join(s.MODEL_DIR, model) + ".p", "rb") as handle:
        pred_result = pickle.load(handle)
    classifier = is_classifier(pred_result["model"])
    regressor = is_regressor(pred_result["model"])
    assert classifier or regressor


def test_train_dt():
    """
    Test whether decision tree is trained and can be loaded.
    """
    model = "dt"
    train(model, DATASET)
    with open(os.path.join(s.MODEL_DIR, model) + ".p", "rb") as handle:
        pred_result = pickle.load(handle)
    classifier = is_classifier(pred_result["model"])
    regressor = is_regressor(pred_result["model"])
    assert classifier or regressor
