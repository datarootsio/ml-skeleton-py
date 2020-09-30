import os

import pandas as pd
import pytest

from ml_skeleton_py import settings as s
from ml_skeleton_py.etl.generate_dataset import generate
from ml_skeleton_py.etl.generate_dataset import remove_outliers

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPECTED_HEADERS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "Class",
]

EXPECTED_N_HEADERS = len(EXPECTED_HEADERS)
UNEXPECTED_N_HEADERS = len(EXPECTED_HEADERS) - 10


@pytest.mark.parametrize(
    "raw_data_loc, transformed_data_loc, error_expected",
    [
        (s.EXPECTED_RAW_DATA_LOC, s.EXPECTED_TEMP_TRANSFORMED_DATA_LOC, False),
        (s.UNEXPECTED_RAW_DATA_LOC, s.EXPECTED_TEMP_TRANSFORMED_DATA_LOC, True),
    ],
)
def test_generate(
        raw_data_loc: str, transformed_data_loc: str, error_expected: bool
) -> None:
    """
    Tests whether file can be opened from ./data/raw
    """
    try:
        generate(raw_data_loc, transformed_data_loc)

    except FileNotFoundError:
        assert error_expected


@pytest.mark.parametrize(
    "dataset_loc, number_of_cols, error_expected",
    [
        (s.EXPECTED_TEMP_TRANSFORMED_DATA_LOC, EXPECTED_N_HEADERS, False),
        (s.EXPECTED_TEMP_TRANSFORMED_DATA_LOC, UNEXPECTED_N_HEADERS, True),
        (s.EXPECTED_TEMP_TRANSFORMED_DATA_LOC, EXPECTED_N_HEADERS, True),
    ],
)
def test_transformed_df(
        dataset_loc: str, number_of_cols: int, error_expected: bool
) -> None:
    """
    Tests whether file can be opened from ./data/transformed and it has the
    desired number of rows, columns and headers
    """
    try:
        df = pd.read_csv(dataset_loc)
        assert df.shape[1] == number_of_cols  # columns
        assert df.shape[0] > 0  # Rows
        assert df.columns.to_list() == EXPECTED_HEADERS
    except AssertionError:
        assert error_expected
    except FileNotFoundError:
        assert error_expected


@pytest.mark.parametrize(
    "dataset_loc, number_of_cols, error_expected",
    [
        (s.EXPECTED_TEMP_TRANSFORMED_DATA_LOC, EXPECTED_N_HEADERS, False),
        (s.EXPECTED_TEMP_TRANSFORMED_DATA_LOC, UNEXPECTED_N_HEADERS, True),
    ],
)
def test_remove_outliers(
        dataset_loc: str, number_of_cols: int, error_expected: bool
) -> None:
    """
    Tests whether outlier removal creates unexpected behaviors. Make sure
    the outlier removed data frame has exactly the same headers with the
    original data frame and have less than or equal to number of rows.
    """
    try:
        df = pd.read_csv(dataset_loc)
        df_outlier_removed = remove_outliers(df)
        assert df.shape[1] == number_of_cols  # columns
        assert df.shape[0] > 0  # Rows
        assert df.columns.to_list() == EXPECTED_HEADERS
        assert df.shape[1] == df_outlier_removed.shape[1]
        assert df.shape[0] >= df_outlier_removed.shape[0]
    except AssertionError:
        assert error_expected
