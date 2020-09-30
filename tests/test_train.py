import os

import joblib
import pytest
from sklearn.base import is_classifier, is_regressor

from ml_skeleton_py import settings as s
from ml_skeleton_py.model import train


@pytest.mark.parametrize(
    "dataset_loc, model_dir, model_name, error_expected",
    [
        (s.EXPECTED_TRANSFORMED_DATA_LOC, s.EXPECTED_MODEL_LOC, "temp_model", False),
        (s.UNEXPECTED_TRANSFORMED_DATA_LOC, s.EXPECTED_MODEL_LOC, "temp_model", True),
    ],
)
def test_train(
        dataset_loc: str, model_dir: str, model_name: str, error_expected: bool
) -> None:
    """
    Test whether logistic regression is trained and can be loaded.
    """
    try:
        train(dataset_loc, model_dir, model_name)
        model_loc = os.path.join(model_dir, model_name) + ".joblib"
        model = joblib.load(model_loc)
        sk_classifier = is_classifier(model["deserialized_model"])
        sk_regressor = is_regressor(model["deserialized_model"])
        assert sk_classifier or sk_regressor
        assert isinstance(model["model_name"], str)
        assert isinstance(model["roc_auc"], float)
    except FileNotFoundError:
        assert error_expected
