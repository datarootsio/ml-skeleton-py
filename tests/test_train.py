import os
from sklearn.base import is_classifier, is_regressor
from ml_skeleton_py import settings as s
from ml_skeleton_py.model.train import train
from ml_skeleton_py.helper import load_pickle

DATASET = "creditcard.csv"
MODEL_NAME = "lr_test"


def test_train_lr() -> None:
    """
    Test whether logistic regression is trained and can be loaded.
    """
    train(DATASET, MODEL_NAME)
    pred_result = load_pickle(os.path.join(s.MODEL_DIR, MODEL_NAME) + ".p")
    classifier = is_classifier(pred_result["model"])
    regressor = is_regressor(pred_result["model"])
    assert classifier or regressor
