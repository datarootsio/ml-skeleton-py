import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import settings as s
from src.etl.generate_dataset import generate, remove_outliers


df = generate()


def test_generate():
    """
    Tests whether file can be opened and whether it is a pd.DataFrame.
    """
    df = pd.read_csv(os.path.join(s.DATA_TRANSFORMED, "creditcard.csv"))
    assert type(df) == pd.DataFrame


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


# def test_generate_loading_X_train():
#     """
# 	test if X_train is loaded correctly
# 	"""
#     X_train_load = pickle.load(
#         open(os.path.join(s.DATA_TRANSFORMED, "X_train.p"), "rb")
#     )
#     assert (X_train_load == X_train).all()


# def test_generate_loading_y_train():
#     """
# 	test if y_train is loaded correctly
# 	"""
#     y_train_load = pickle.load(
#         open(os.path.join(s.DATA_TRANSFORMED, "y_train.p"), "rb")
#     )
#     assert (y_train_load == y_train).all()


# def test_generate_loading_X_test():
#     """
# 	test if X_test is loaded correctly
# 	"""
#     X_test_load = pickle.load(open(os.path.join(s.DATA_TRANSFORMED, "X_test.p"), "rb"))
#     assert (X_test_load == X_test).all()


# def test_generate_loading_y_test():
#     """
# 	test if y_test is loaded correctly
# 	"""
#     y_test_load = pickle.load(open(os.path.join(s.DATA_TRANSFORMED, "y_test.p"), "rb"))
#     assert (y_test_load == y_test).all()
