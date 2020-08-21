import os
from ml_skeleton_py import settings as s
from ml_skeleton_py.model.train import train
from ml_skeleton_py.model.predict import predict
from ml_skeleton_py.model.predict import predict_from_file
import pandas as pd

DATASET = "creditcard.csv"
MODEL_NAME = "lr_test"


def test_predict_1() -> None:
    """
    Test whether an observation makes a prediction.
    """
    train(MODEL_NAME, DATASET)
    model_name = "lr_test.p"
    observation = [
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
    prediction = predict([observation], model_name)
    assert type(float(prediction)) == float


def test_predict_2() -> None:
    """
    Test whether an observation makes a prediction.
    """
    model_name = "lr_test.p"
    observation = [
        0.79754226,
        1.06912246,
        -0.68417044,
        -1.16227508,
        -0.99710428,
        -0.31829266,
        -1.22540427,
        -0.12708047,
        -1.30367552,
        -1.65065419,
        1.84231034,
        -0.54632697,
        -0.82522528,
        0.68983756,
        0.05942017,
        -0.26308571,
        -1.13504082,
        0.67475109,
        -0.38479205,
        -0.06393738,
        -1.08399423,
        0.01850012,
        0.43097685,
        0.01968036,
        -0.14525213,
        0.54279049,
        0.24565009,
        -0.30732591,
        -0.67713242,
        -0.20959966,
    ]
    prediction = predict([observation], model_name)
    assert type(float(prediction)) == float


def test_predict_3() -> None:
    """
    Test whether an observation makes a prediction.
    """
    model_name = "lr_test.p"
    observation = [
        -0.14526721,
        -1.88587432,
        0.09173321,
        -2.13679426,
        2.47527663,
        -1.39709787,
        -1.58094806,
        -3.24989715,
        3.10746685,
        -1.72422133,
        -4.64074371,
        2.93569031,
        -5.02776476,
        -0.98484625,
        -6.79537872,
        0.77480618,
        -3.41292719,
        -6.93117153,
        -1.03713254,
        1.34314323,
        3.09149758,
        2.0029368,
        -0.17721712,
        -1.38678516,
        0.31678544,
        -0.24190815,
        -0.35145657,
        7.70963483,
        -1.08784478,
        3.03081115,
    ]
    prediction = predict([observation], model_name)
    assert type(float(prediction)) == float


def test_predict_from_file_balanced() -> None:
    """
    Test whether predict from file works properly
    """
    file_name = "test_balanced_predictions.csv"
    predict_from_file("lr_test.p", "test_balanced_creditcard.csv", file_name)

    pred_path = os.path.join(s.DATA_PREDICTIONS, file_name)
    df = pd.read_csv(pred_path, header=None)
    assert type(df) == pd.DataFrame


def test_predict_from_file_imbalanced() -> None:
    """
    Test whether predict from file works properly
    """
    file_name = "test_imbalanced_predictions.csv"
    predict_from_file("lr_test.p", "test_imbalanced_creditcard.csv", file_name)

    pred_path = os.path.join(s.DATA_PREDICTIONS, file_name)
    df = pd.read_csv(pred_path, header=None)
    assert type(df) == pd.DataFrame
