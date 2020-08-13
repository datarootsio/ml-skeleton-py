"""Predict example.

Example borrowed from:
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
"""

import os
import time
import logging
import functools
import pickle

# from sklearn.externals import joblib
import joblib
import numpy as np

# from mlmonkey.metadata import PredictionMetadata

from .. import settings as s

logger = logging.getLogger(__name__)


# @functools.lru_cache()
# def load_model(model_location: str):
#     """Load model using joblib.

#     Uses lru for caching.

#     :param model_location: path to model file
#     """
#     return joblib.load(model_location)


def predict_from_file(model_name: str, input_df: str, output_df: str) -> None:
    """Predict new values using a serialized model.

    Note log predictions via logger to STDOUT. This should be captured by a
    listener. If not, make amends.

    :param model_name: name find the loaded model by (excluding extension)
    :param input_df: the input features to use to generate prediction on
        (ignored in this example)
    :param output_df: the output data file to store predictions as
    """
    # Deserialize the model
    logger.info("deserializing model: {}".format(model_name))
    # model_location = os.path.join(s.MODEL_DIR, '{}.joblib'.format(model_name))
    # model = load_model(model_location)

    model = pickle.load(open(os.path.join(s.MODEL_DIR, model_name), "rb"))["model"]

    # input features - normally one would load a file based on the `input_df` path here
    # input_data = np.array([
    #     [5.8, 2.8, 2.4],
    #     [6.4, 2.8, 2.1]
    # ])
    input_data = pickle.load(open(os.path.join(s.DATA_TRANSFORMED, input_df), "rb"))

    # only log this directly when batch is small-ish or when predicting for
    # single observations at a time
    logger.info("running predictions for input: {}".format(input_data))

    preds = model.predict(input_data)
    preds = preds.reshape(-1, 1)  # transform single axis array to a column
    # print(preds)

    # only log this directly when batch is small-ish or when predicting for
    # single observations at a time
    logger.info("prediction results: {}".format(preds))

    output_df_fn = os.path.join(s.DATA_PREDICTIONS, output_df)
    logger.info("storing saved prediction at: {}".format(output_df_fn))
    np.savetxt(output_df_fn, preds, delimiter=",")

    # pm = PredictionMetadata(model_location=model_location,
    #                         input_identifier=input_data.tolist(),
    #                         output_identifier=preds.tolist())

    # logger.info('prediction base metadata: {}'.format(pm))


def predict_api(body: str, model_name: str) -> dict:
    """
    Make prediction through an API call.

    :param body: the body of the request
    :param model_name: the name of the model, will be used to deserialize the model

    :return: predictions
    """
    start_time = time.time()
    features = np.array(body.get("features"))

    logger.info("deserializing model: {}".format(model_name))

    # deserialize the model

    model_location = os.path.join(s.MODEL_DIR, "{}.joblib".format(model_name))
    model_metadata_location = os.path.join(
        s.MODEL_METADATA_DIR, "{}.joblib.json".format(model_name)
    )

    model = load_model(model_location)

    logger.info("running predictions for input: {}".format(body))

    preds = model.predict(features)
    preds = preds.reshape(-1, 1)  # transform single axis array to a column

    logger.info("prediction results: {}".format(preds))

    return {
        "release": {
            "model_name": model_name,
            "model_location": model_location,
            "model_metadata_location": model_metadata_location,
        },
        "result": preds.tolist(),
        "timing": time.time() - start_time,
    }
