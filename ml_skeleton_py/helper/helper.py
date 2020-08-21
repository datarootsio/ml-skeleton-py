"""
This file contains some helper functions.
"""

import pickle
from typing import Any


def save_pickle(object: Any, save_path: str) -> None:
    """
    Save pickle object in a specific directory.

    Parameters:
        pred_result (Any): object that you want to save

        save_path (str): path on which you want to save a file

    Returns:
        None
    """
    with open(save_path, "wb") as handle:
        pickle.dump(object, handle)
