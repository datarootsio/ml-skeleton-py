import pytest
from ml_skeleton_py.etl.generate_dataset import generate, generate_test
import pandas as pd
from ml_skeleton_py.model.train import train

DATASET = "creditcard.csv"
MODEL_NAME = "lr_test"


@pytest.fixture(scope="module")
def test_generate_df() -> pd.DataFrame:
    """
    Generate creditcard.csv dataset.
    """
    df = generate(DATASET)
    return df


@pytest.fixture(scope="module")
def test_generate_test_df() -> None:
    """
    Generate test sets on the basis of creditcard.csv dataset.
    """
    generate_test(DATASET)


@pytest.fixture(scope="module")
def test_train() -> None:
    """
    Train logistic regression on creditcard.csv dataset.
    """
    train(DATASET, MODEL_NAME)
