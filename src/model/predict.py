"""Predict example.

Example borrowed from:
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
"""

import os
import json
import logging

from sklearn.externals import joblib
import pandas as pd
import click

from mlmonkey.metadata import PredictionMetadata
from .. import settings as s

logger = logging.getLogger(__name__)


@click.command()
@click.option('--model-name', default='model')
@click.option('--input-df', default='model')
@click.option('--output-df', default='predictions.csv')
def main(model_name, input_df, output_df):
    """Predict new values using a serialized model.

    Note log predictions via logger to STDOUT. This should be captured by a
    listener. If not, make amends.

    :param model_name: name find the loaded model by (excluding extension)
    :param input_df: the input features to use to generate prediction on
        (ignored in this example)
    :param output_df: the output data file to store predictions as
    """
    # Deserialize the model
    logger.info('deserializing model: {}'.format(model_name))
    model_location = os.path.join(s.MODEL_DIR, '{}.p'.format(model_name))
    regr = joblib.load(model_location)

    # Run prediction
    input_data = [
        [5.8, 2.8, 2.4],
        [6.4, 2.8, 2.1]
    ]

    # only log this directly when batch is small-ish or when predicting for
    # single observations at a time
    logger.info('running predictions for input: {}'.format(input_data))

    preds = regr.predict(input_data)
    preds_df = pd.DataFrame({'predictions': preds})

    # only log this directly when batch is small-ish or when predicting for
    # single observations at a time
    logger.info('prediction results: {}'.format(preds))

    output_df_fn = os.path.join(s.DATA_PREDICTIONS, output_df)
    logger.info('storing saved prediction at: {}'.format(output_df_fn))
    preds_df.to_csv(output_df_fn, index=False)

    pm = PredictionMetadata(model_location=model_location,
                            input_identifier=input_data,
                            output_identifier=preds.tolist())

    logger.info('prediction base metadata: {}'.format(
        json.dumps(pm.get())))


if __name__ == '__main__':
    main()
