"""Example how to start Flask endpoint, to serve the predictions."""
import numpy
import os
import time

import click
from flask import Flask, request, jsonify
import logging

from src import settings
from sklearn.externals import joblib

logger = logging.getLogger(__name__)
app = Flask(__name__)
model = None


@app.route('/predict', methods=['POST'])
def predict():
    """Return prediction for given request."""
    try:
        data = request.get_json()
        features = data['features']

        if not isinstance(features, list):
            return jsonify('Features parametar must be a list of entries. '
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
    assert isinstance(features, list)
    np_features = numpy.array(features)
    if np_features.ndim == 1:
        np_features = np_features.reshape(1, -1)

    start_time = time.time()
    prediction = list(model.predict(np_features))
    model_id = model._custom_metadata_ids['model_identifier']
    model_git_commit = model._custom_metadata_ids['git_commit']
    elapsed_time = time.time() - start_time

    return {
        'release': {
            'model_id': model_id,
            'git_commit': model_git_commit
        },
        'result': prediction,
        'timing': elapsed_time
    }


@click.command()
@click.option('--model-path',
              default=os.path.join(settings.MODEL_DIR, '{}.p'.format('model')))
@click.option('--host', default='localhost')
@click.option('--port', default=5000)
def main(host, port, model_path):
    """Load model and start flask app.

    :param model_path: path to serialized model
    :param host: server host
    :param port: port
    """
    logger.info('Deserializing model: {}'.format(model_path))
    global model
    model = joblib.load(model_path)

    logger.info('Starting flask server...')
    app.run(host=host, port=port, debug=True)


if __name__ == '__main__':
    main()