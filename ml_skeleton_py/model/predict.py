"""Predict example.

Example borrowed from:
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
"""

import functools
import os

import joblib
import numpy as np
import sklearn.pipeline


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
        model = joblib.load(handle)["model"]
    return model


# @dploy endpoint predict
def predict(body: dict) -> float:
    """
    Predict one single observation.

    Parameters:
        body (dict): having the and model name and features. Model name is the serialized model name
                     in string format. Features represents all the features in list type that will
                     be used to do the predictions
                     I.e {"model_name": "lr.joblib",
                          "features": [0.12, 0.56, ..., 0.87]}
    Return:
        prediction (float): the prediction
    """
    model = load_model(body["model_name"])
    features = np.asarray(body["features"]).reshape(1, -1)
    return float(model.predict(features)[0])
