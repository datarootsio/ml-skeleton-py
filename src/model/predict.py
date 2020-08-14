"""Predict example.

Example borrowed from:
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
"""

import os
import time
import logging
import functools
import pickle
from sklearn.base import BaseEstimator

# from sklearn.externals import joblib
import numpy as np

# from mlmonkey.metadata import PredictionMetadata

# from .. import settings as s

from config import settings as s

logger = logging.getLogger(__name__)


@functools.lru_cache()
def load_model(model_name: str) -> BaseEstimator:
    """
    Load model using pickle.

    Uses lru for caching.

    Parameters:
        model_name (str): model name e.g. "lr.p"

    Returns:
        model (Pipeline or BaseEstimator): a model that can make predictions
    """
    with open(os.path.join(s.MODEL_DIR, model_name), "rb") as handle:
        model = pickle.load(handle)["model"]
    return model


def predict(model_name: str, observation: np.array) -> float:
    """
    Predict one single observation.

    Parameters:
        model_name (str): the name of the model file that you want to use

        observation (np.array): the variables of the input observation

    Return:
        prediction (float): the prediction
    """
    model = load_model(model_name)
    prediction = next(iter(model.predict(observation)))
    return prediction


def predict_from_file(model_name: str, input_df: str, output_df: str) -> np.array:
    """Predict new values using a serialized model.

    Note log predictions via logger to STDOUT. This should be captured by a
    listener. If not, make amends.

    Parameters:
        model_name (str): name find the model to load (including extension)

        input_df (str): the input features to use to generate prediction on

        output_df (str): the output data file to store predictions as

    Returns:
        preds (np.array): predictions
    """

    # Deserialize the model
    logger.info("deserializing model: {}".format(model_name))

    # load input_data
    with open(os.path.join(s.DATA_TRANSFORMED, input_df), "rb") as handle:
        input_data = pickle.load(handle)

    # only log this directly when batch is small-ish or when predicting for
    # single observations at a time
    logger.info("running predictions for input: {}".format(input_data))

    # make predictions
    preds = [predict(model_name, [x]) for x in input_data]
    preds = np.array(preds).reshape(-1, 1)  # transform single axis array to a column

    # only log this directly when batch is small-ish or when predicting for
    # single observations at a time
    logger.info("prediction results: {}".format(preds))

    # save the predictions
    output_df_fn = os.path.join(s.DATA_PREDICTIONS, output_df)
    logger.info("storing saved prediction at: {}".format(output_df_fn))
    np.savetxt(output_df_fn, preds, delimiter=",")

    return preds


