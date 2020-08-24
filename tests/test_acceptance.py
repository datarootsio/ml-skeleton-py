import os
from ml_skeleton_py import settings as s
from ml_skeleton_py.model.train import train
from ml_skeleton_py.helper import load_pickle

DATASET = "creditcard.csv"
MODEL_NAME = "lr_acceptance"


def test_acceptance() -> None:
    """
    Test whether logistic regression passes roc auc of 0.85.
    """
    train(DATASET, MODEL_NAME)
    pred_result = load_pickle(os.path.join(s.MODEL_DIR, MODEL_NAME) + ".p")
    assert pred_result["roc_auc"] > 0.85
