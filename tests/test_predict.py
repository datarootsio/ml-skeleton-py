import os

import numpy as np
import pytest

from ml_skeleton_py import settings as s
from ml_skeleton_py.model import train
from ml_skeleton_py.model.predict import load_model

features_1 = [
    -0.51056756,
    -4.76915766,
    4.17380769,
    -6.18019076,
    5.54479825,
    -6.07673393,
    -2.83891627,
    -12.14473542,
    11.95168444,
    -5.89969894,
    -12.93298794,
    4.58542528,
    -13.04122239,
    0.80026314,
    -15.05300726,
    0.80569352,
    -11.45602963,
    -23.21915935,
    -7.54677977,
    3.40316942,
    0.04731062,
    6.27192486,
    0.1867837,
    -5.35273187,
    0.65159854,
    -0.06661776,
    0.71556094,
    1.68012583,
    -1.25077894,
    -0.30741284,
]

features_2 = [
    -0.51056756,
    -4.76915766,
    4.17380769,
    -6.18019076,
    5.54479825,
    -6.07673393,
    -2.83891627,
    -12.14473542,
    11.95168444,
    -5.89969894,
    -12.93298794,
    4.58542528,
    -13.04122239,
    0.80026314,
    -15.05300726,
    0.80569352,
    -11.45602963,
    -23.21915935,
    -7.54677977,
    3.40316942,
    0.04731062,
    6.27192486,
    0.1867837,
    -5.35273187,
    0.65159854,
    -0.06661776,
    0.71556094,
    1.68012583,
]

features_3 = [
    -0.51056756,
    -4.76915766,
    4.17380769,
    -6.18019076,
    5.54479825,
    None,
    None,
    None,
    11.95168444,
    -5.89969894,
    -12.93298794,
    4.58542528,
    -13.04122239,
    0.80026314,
    -15.05300726,
    0.80569352,
    -11.45602963,
    -23.21915935,
    -7.54677977,
    3.40316942,
    0.04731062,
    6.27192486,
    0.1867837,
    -5.35273187,
    0.65159854,
    -0.06661776,
    0.71556094,
    1.68012583,
    -1.25077894,
    -0.30741284,
]

# Need to train first to test predict
train(s.EXPECTED_TRANSFORMED_DATA_LOC, s.EXPECTED_MODEL_LOC, "test_model")


@pytest.mark.parametrize(
    "features, error_expected",
    [(features_1, False), (features_2, True), (features_3, True)],
)
def test_pred(features: list, error_expected: bool) -> None:
    """
    Test whether an observation makes a prediction.
    """
    model_loc = os.path.join(s.EXPECTED_MODEL_LOC, "test_model.joblib")
    model = load_model(model_loc)
    features = np.asarray(features).reshape(1, -1)
    try:
        assert model.predict(features)[0] >= 0
    except ValueError:
        assert error_expected
