import os
from sklearn.base import is_classifier, is_regressor
from ml_skeleton_py import settings as s
from ml_skeleton_py.helper import load_pickle
import pandas as pd

MODEL_NAME = "lr_test"


def test_train_lr(test_generate_df: pd.DataFrame, test_train: None) -> None:
    """
    Test whether logistic regression is trained and can be loaded.
    """
    pred_result = load_pickle(os.path.join(s.MODEL_DIR, MODEL_NAME) + ".p")
    classifier = is_classifier(pred_result["model"])
    regressor = is_regressor(pred_result["model"])
    assert classifier or regressor
