"""Predict example.
"""

import functools
import os
from typing import Dict, Union, List

import joblib
import numpy as np
import sklearn.pipeline

from ml_skeleton_py import settings as s


@functools.lru_cache()
def load_model(model_loc: str) -> sklearn.pipeline:
    """
    Load models using joblib.

    Uses lru for caching.

    Parameters:
        model_loc (str): models file name e.g. "lr.joblib"

    Returns:
        models (Pipeline or BaseEstimator): sk-learn fitted models
    """
    return joblib.load(model_loc)["deserialized_model"]


# Don't delete the following line, it is required
# to deploy this method via dploy-kickstart
# @dploy endpoint predict
def predict(body: Dict[str, Union[str, List[float]]]) -> float:
    """
    Predict one single observation.

    Parameters:
        body (dict): having the serialized model name (str)
                     and features (list).
                     I.e {"model_f_name": "lr.joblib",
                          "features": [0.12, 0.56, ..., 0.87]}
    Return:
        prediction (float): the prediction
    """
    model_loc = os.path.join(s.MODEL_DIR, body["model_f_name"])
    model = load_model(model_loc)
    features = np.asarray(body["features"]).reshape(1, -1)
    return float(model.predict(features)[0])
