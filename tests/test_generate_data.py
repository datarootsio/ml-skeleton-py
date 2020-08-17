import os
import sys
import pandas as pd

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml_skeleton_py import settings as s
from ml_skeleton_py.etl.generate_dataset import generate, remove_outliers


DATASET = "creditcard.csv"
df = generate(DATASET)

def test_generate():
    """
    Tests whether file can be opened and whether it is a pd.DataFrame.
    """
    df = pd.read_csv(os.path.join(s.DATA_TRANSFORMED, DATASET))
    assert type(df) == pd.DataFrame

def test_faulty_generate():
    """
    Test whether non existent file is returned with None.
    """
    assert generate("no.csv") is None

def test_outlier_removal():
    """
    Tests whether outlier removal works as intended.
    """
    params = {
        "V14_upper": 3.8320323237414122,
        "V14_lower": -17.807576138200663,
        "V12_upper": 5.597044719256134,
        "V12_lower": -17.25930926645337,
        "V10_upper": 5.099587558797303,
        "V10_lower": -15.47046969983434,
    }
    assert type(remove_outliers(df, params)) == pd.DataFrame
