#!/usr/bin/env python3

import os
import sys
import functools

import connexion

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.predict import predict_api

predict_func = functools.partial(predict_api, model_name='model')

if __name__ == '__main__':
    app = connexion.FlaskApp(__name__, port=9090, specification_dir='../openapi/', debug=True)
    app.add_api('prediction-api.yaml', arguments={'model_name': 'model'})
    app.run(host='0.0.0.0')