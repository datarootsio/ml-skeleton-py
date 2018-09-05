## example borrowed from
## # http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

import os
import json
import logging

from sklearn.externals import joblib
import pandas as pd
import click

from ..helpers import base_metadata

logger = logging.getLogger(__name__)

@click.command()
@click.option('--model-name', default='model')
@click.option('--input-df', default='model')
@click.option('--output-df', default='predictions.csv')
def main(model_name, input_df, output_df):
    """
    Predict new values using a serialized model

    Note log predictions via logger to STDOUT. This should be captured by a listener. If not, make amends.

    :param model_name: name find the loaded model by (excluding extension)
    :param input_df: the input features to use to generate prediction on (ignored in this example)
    :param output_df: the output data file to store predictions as
    """

    # Deserialize the model
    logger.info('deserializing model: {}'.format(model_name))
    regr = joblib.load(os.path.join(os.getenv('MODEL_DIR'), '{}.p'.format(model_name)))

    # Run prediction
    input_data = [
        [5.8, 2.8, 2.4],
        [6.4, 2.8, 2.1]
    ]

    # only log this directly when batch is small-ish or when predicting for single observations at a time
    logger.info('running predictions for input: {}'.format(input_data))

    preds = regr.predict(input_data)
    preds_df = pd.DataFrame({'predictions':preds})

    # only log this directly when batch is small-ish or when predicting for single observations at a time
    logger.info('prediction results: {}'.format(preds))

    output_df_fn = os.path.join(os.getenv('DATA_PREDICTIONS'), output_df)
    logger.info('storing saved prediction at: {}'.format(output_df_fn))

    logger.info('prediction base metadata: {}'.format(json.dumps(base_metadata())))
    preds_df.to_csv(output_df_fn, index=False)


if __name__ == '__main__':
    main()