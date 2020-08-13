import os
import sys
from sklearn.base import is_classifier, is_regressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import settings as s
from src.model.train import fetch_model
from src.model.train import train
import pickle


def test_lr():
    """
    Test logistic regression call for model.
    """
    model = "lr"
    classifier, params = fetch_model(model)
    assert (is_classifier(classifier) or is_regressor(classifier)) and (
        type(params) == dict
    )


def test_knn():
    """
    Test logistic regression call for model.
    """
    model = "knn"
    classifier, params = fetch_model(model)
    assert (is_classifier(classifier) or is_regressor(classifier)) and (
        type(params) == dict
    )


def test_svc():
    """
    Test logistic regression call for model.
    """
    model = "svc"
    classifier, params = fetch_model(model)
    assert (is_classifier(classifier) or is_regressor(classifier)) and (
        type(params) == dict
    )


def test_dt():
    """
    Test logistic regression call for model.
    """
    model = "dt"
    classifier, params = fetch_model(model)
    assert (is_classifier(classifier) or is_regressor(classifier)) and (
        type(params) == dict
    )


def test_train_lr():
    """
    Test whether a model is trained and a model can be loaded.
    """
    train()
    model = "lr"
    with open(os.path.join(s.MODEL_DIR, model) + ".p", "rb") as handle:
        pred_result = pickle.load(handle)
    assert is_classifier(pred_result["model"]) or is_regressor(pred_result["model"])
