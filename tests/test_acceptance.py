import os
from ml_skeleton_py import settings as s
from ml_skeleton_py.helper import load_pickle
import pandas as pd

MODEL_NAME = "lr_test"


def test_acceptance(test_generate_df: pd.DataFrame, test_train: None) -> None:
    """
    Test whether logistic regression passes roc auc of 0.85.
    """
    pred_result = load_pickle(os.path.join(s.MODEL_DIR, MODEL_NAME) + ".p")
    assert pred_result["roc_auc"] > 0.85
