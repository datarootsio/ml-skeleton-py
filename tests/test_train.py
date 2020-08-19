import os
from sklearn.base import is_classifier, is_regressor
from ml_skeleton_py import settings as s
from ml_skeleton_py.model.train import train
import pickle

DATASET = "creditcard.csv"
MODEL_NAME = "lr_test"


def test_train_lr() -> None:
    """
    Test whether logistic regression is trained and can be loaded.
    """
    train(MODEL_NAME, DATASET)
    with open(os.path.join(s.MODEL_DIR, MODEL_NAME) + ".p", "rb") as handle:
        pred_result = pickle.load(handle)
    classifier = is_classifier(pred_result["model"])
    regressor = is_regressor(pred_result["model"])
    assert classifier or regressor
