"""Predict example.

Example borrowed from:
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
"""

import functools
import os

import joblib
import numpy as np
import sklearn.pipeline

from ml_skeleton_py import settings as s


@functools.lru_cache()
def load_model(model_loc: str) -> sklearn.pipeline:
    """
    Load models using pickle.

    Uses lru for caching.

    Parameters:
        model_loc (str): models file name e.g. "lr.joblib"

    Returns:
        models (Pipeline or BaseEstimator): a models that can make predictions
    """
    return joblib.load(model_loc)["deserialized_model"]


# Don't delete the following line, it is required to deploy this function via dploy-kickstart
# @dploy endpoint predict
def predict(body: dict) -> float:
    """
    Predict one single observation.

    Parameters:
        body (dict): having the models name and features. Model name is the serialized models name
                     in string type. Features represent all the features in list type that will
                     be used to do the predictions
                     I.e {"model_f_name": "lr.joblib",
                          "features": [0.12, 0.56, ..., 0.87]}
    Return:
        prediction (float): the prediction
    """
    model_loc = os.path.join(s.MODEL_DIR, body["model_f_name"])
    model = load_model(model_loc)
    features = np.asarray(body["features"]).reshape(1, -1)
    return float(model.predict(features)[0])
