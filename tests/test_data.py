import os
import numpy as np
import pandas as pd

from ml_skeleton_py import settings as s


def test_data() -> None:
    """
    Test whether there are missing values.
    """
    df = pd.read_csv(os.path.join(s.DATA_RAW, "creditcard.csv"))
    assert not (True in np.isnan(df))
