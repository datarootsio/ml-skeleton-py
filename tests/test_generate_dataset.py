import pandas as pd
import pytest

from ml_skeleton_py.etl.generate_dataset import remove_outliers

DF_EXPECTED = pd.DataFrame(
    {
        "V10": [-1, -1.5, 100, -2],
        "V12": [20, 21, 600, 18],
        "V14": [20, 21, 10000, 18],
    }
)

DF_UNEXPECTED_1 = pd.DataFrame(
    {
        "V10": [-1, -1.5, 100, -2],
        "V12": [20, 21, 600, 18],
        "test": [20, 21, 20, 18],
    }
)

DF_UNEXPECTED_2 = pd.DataFrame(
    {
        "V10": [-1, None, 100, -2],
        "V12": [20, 21, 600, 18],
        "V14": [20, 21, 20, 18],
    }
)


@pytest.mark.parametrize(
    "df, error_expected",
    [
        (DF_EXPECTED, False),
        (DF_UNEXPECTED_1, True),
        (DF_UNEXPECTED_2, True),
    ],
)
def test_remove_outliers(df: pd.DataFrame, error_expected: bool) -> None:
    """
    Tests whether outlier removal creates unexpected behaviors. Make sure
    the outlier removed data frame has exactly the same headers with the
    original data frame and have less than or equal to number of rows.
    """
    try:
        df_outlier_removed = remove_outliers(df, n_neighbors=2)
        assert df.shape[0] > 0  # Rows
        assert df.shape[1] == df_outlier_removed.shape[1]
        assert df.shape[0] > df_outlier_removed.shape[0]
    except KeyError:
        assert error_expected
    except ValueError:
        assert error_expected
