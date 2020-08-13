import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.settings as s
import pickle


df = pd.read_csv(os.path.join(s.DATA_RAW, "creditcard.csv"))


def test_data():
    assert 1 == 1
