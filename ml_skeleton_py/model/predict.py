"""Predict example.

Example borrowed from:
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
"""

import os
import functools
import pickle
import sklearn.pipeline
import numpy as np


@functools.lru_cache()
def load_model(model_name: str) -> sklearn.pipeline:
    """
    Load model using pickle.

    Uses lru for caching.

    Parameters:
        model_name (str): model name (including extension) e.g. "lr.p"

    Returns:
        model (Pipeline or BaseEstimator): a model that can make predictions
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root_dir, "models", model_name), "rb") as handle:
        model = pickle.load(handle)["model"]
    return model


# @dploy endpoint predict
def predict(observation: np.array, model_name: str = "lr.p") -> float:
    """
    Predict one single observation.

    Parameters:
        observation (np.array): the input observation

        model_name (str): the name of the model file that you want to load
                          (including extension)
                          default value = "lr.p"

    Return:
        prediction (float): the prediction
    """
    model = load_model(model_name)
    return float(model.predict(observation)[0])
