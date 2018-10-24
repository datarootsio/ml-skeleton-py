"""Example how to start Flask endpoint, to serve the predictions."""

import os
import sys
import time

from flask import Flask, request, jsonify
import logging

from src import settings
from src.helpers import most_recent_model_id
from sklearn.externals import joblib

logger = logging.getLogger(__name__)
app = Flask(__name__)


@app.route('/predict')
def predict():
    """Return prediction for given request."""
    try:
        data = request.get_json()
        features = data['features']

        if not isinstance(features, list):
            return jsonify('Features parametar must be a list of entries.'
                           'Each entry is a list of feature values')

        logger.info('Calculating prediction for input data: {}'
                    .format(features))
        response = generate_response(model, features)

        logger.info('Response: {}'.format(response))
    except Exception as e:
        return jsonify('Error occurred. {}'.format(str(e)))

    return jsonify(response)


def generate_response(model, features):
    """
    Calculate predictions for given model and features and generate response.

    :param model: Predictive model
    :param features: List of entries for which to compute predictions.
    Each entry is a list of feature values.
    :return: Response from api endpoint.
    """
    start_time = time.time()
    prediction = list(model.predict(features))
    model_id = most_recent_model_id()
    elapsed_time = time.time() - start_time

    return {
        'model_id': model_id,
        'prediction': prediction,
        'timing': elapsed_time
    }


if __name__ == '__main__':
    # Name of the model is given as a second argument
    # (script name is the first argument always).
    if len(sys.argv) == 1:
        model_name = 'model'
    else:
        model_name = sys.argv[1]

    logger.info('Deserializing model: {}'.format(model_name))
    model = joblib.load(os.path.join
                        (settings.MODEL_DIR, '{}.p'.format(model_name)))

    logger.info('Starting flask server...')
    app.run(debug=True)
