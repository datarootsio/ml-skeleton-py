"""Example how to start Flask endpoint, to serve the predictions."""

import os
import sys
from flask import Flask, request, jsonify
import logging
from .. import settings as s
from sklearn.externals import joblib

logger = logging.getLogger(__name__)
app = Flask(__name__)


@app.route('/predict')
def predict():
    """Return prediction for given request."""
    try:
        data = request.get_json()
        features = data['features']

        logger.info('Calculating prediction for input data: {}'
                    .format(features))
        prediction = model.predict(features)

        logger.info('Prediction: {}'.format(prediction))
    except Exception:
        return jsonify('Error occurred. Please check your json data format.')

    return jsonify(list(prediction))


if __name__ == '__main__':
    # Name of the model is given as a second argument
    # (script name is the first argument always).
    if len(sys.argv) == 1:
        model_name = 'model'
    else:
        model_name = sys.argv[1]

    logger.info('Deserializing model: {}'.format(model_name))
    model = joblib.load(os.path.join
                        (s.MODEL_DIR, '{}.p'.format(model_name)))

    logger.info('Starting flask server...')
    app.run(debug=True)
