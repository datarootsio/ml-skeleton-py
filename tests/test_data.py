import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import settings as s


df = pd.read_csv(os.path.join(s.DATA_RAW, "creditcard.csv"))


def test_data():
    """
    Test whether there are missing values.
    """
    assert not (True in np.isnan(df))
