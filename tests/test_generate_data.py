import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.settings as s
from src.etl.generate_dataset import generate
import pickle


X_train, X_test, y_train, y_test = generate()


def test_generate_loading_X_train():
    """
	test if X_train is loaded correctly
	"""
    X_train_load = pickle.load(
        open(os.path.join(s.DATA_TRANSFORMED, "X_train.p"), "rb")
    )
    assert (X_train_load == X_train).all()


def test_generate_loading_y_train():
    """
	test if y_train is loaded correctly
	"""
    y_train_load = pickle.load(
        open(os.path.join(s.DATA_TRANSFORMED, "y_train.p"), "rb")
    )
    assert (y_train_load == y_train).all()


def test_generate_loading_X_test():
    """
	test if X_test is loaded correctly
	"""
    X_test_load = pickle.load(open(os.path.join(s.DATA_TRANSFORMED, "X_test.p"), "rb"))
    assert (X_test_load == X_test).all()


def test_generate_loading_y_test():
    """
	test if y_test is loaded correctly
	"""
    y_test_load = pickle.load(open(os.path.join(s.DATA_TRANSFORMED, "y_test.p"), "rb"))
    assert (y_test_load == y_test).all()
