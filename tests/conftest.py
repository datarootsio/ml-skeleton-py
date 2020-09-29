import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_skeleton_py import settings as s

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def create_sample_dataset() -> None:
    """
    Generate sample datasets to easily work on unit tests
    """
    df = pd.read_csv(os.path.join(s.DATA_RAW, s.DATASET_NAME))
    y = df.pop(s.TARGET_VARIABLE)
    _, X_test, _, y_test = train_test_split(df, y, test_size=0.1,
                                            random_state=42, stratify=y)
    X_test['Class'] = y_test
    X_test.to_csv(s.EXPECTED_RAW_DATA_LOC, index=False)


def pytest_sessionstart(session):
    """
    Pytest Hook Method; run this method before running any tests
    """
    create_sample_dataset()
