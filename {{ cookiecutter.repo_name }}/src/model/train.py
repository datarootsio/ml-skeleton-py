## example borrowed from
## # http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

import os
import datetime
import logging

from sklearn import datasets, linear_model
from sklearn.externals import joblib
import pandas as pd
import click

from ..helpers import metadata_to_file

logger = logging.getLogger(__name__)


@click.command()
@click.option('--model-name', default='model')
@click.option('--input-data', default='iris.csv')
def main(model_name, input_data):
    """
    Train and save a linear regression model

    :param model_name: name to save the model as (extension excluded)
    """
    # Load the iris dataset
    logging.info('Loading iris dataset')
    iris = pd.read_csv(os.path.join(os.getenv('DATA_TRANSFORMED'), input_data))
    iris = iris.sample(frac=1)  # shuffle

    # drop species and target (petal_length)
    iris_X = iris.drop(columns=['species', 'petal_length'])

    # Split the data into training/testing sets
    iris_X_train = iris_X[:100]
    iris_X_test = iris_X[100:]

    # Split the targets into training/testing sets
    iris_y_train = iris['petal_length'][:100]
    iris_y_test = iris['petal_length'][100:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    logger.info('Fitting linear model')
    regr.fit(iris_X_train, iris_y_train)

    # save the model
    logger.info('Saving serialized model: {}'.format(model_name))
    joblib.dump(regr,
                os.path.join(os.getenv('MODEL_DIR'), '{}.p'.format(model_name)))

    # save relevant metadata
    logger.info('Saving model metadata for model: {}'.format(model_name))
    metadata_to_file(path=os.getenv('MODEL_METADATA_DIR'),
                     filename=model_name,
                     metadata={'model_name': model_name,
                               'extra': 'some extra information'},
                     logger=logger)


if __name__ == '__main__':
    main()
