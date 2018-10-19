"""Train example. Regression on iris dataset."""

import datetime
import os
import logging
import hashlib

from sklearn import linear_model
from sklearn.externals import joblib
import pandas as pd
import click

from ..helpers import metadata_to_file, get_git_commit
from .. import settings as s
from sklearn.model_selection import KFold, cross_validate

logger = logging.getLogger(__name__)


@click.command()
@click.option('--model-filename', default='model')
@click.option('--input-data', default='iris.csv')
def main(model_filename, input_data):
    """Train and save a model. Calculate evaluation metrics. Write metadata.

    :param model_filename: name of file that stores serialized model
    (without extension)
    :param input_data: name of file that holds data used to train the model
    (with extension)
    """
    # Load the iris dataset
    logging.info('Loading iris dataset')
    data_location = os.path.join(s.DATA_TRANSFORMED, input_data)
    iris = pd.read_csv(data_location)
    iris = iris.sample(frac=1)  # shuffle

    # Prepare train and target columns
    iris_X = iris.drop(columns=['species', 'petal_length'])
    iris_y = iris['petal_length']

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Calculating CV score(s)
    logger.info('Performing cross validation')
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(regr, iris_X, iris_y, cv=cv,
                            scoring=['r2', 'neg_mean_squared_error'],
                            return_train_score=False, verbose=1)
    r2_cv_score = scores['test_r2'].mean()
    # cross_validate outputs negative mse
    mse_cv_score = - scores['test_neg_mean_squared_error'].mean()

    # Train a model using the whole dataset
    logger.info('Fitting linear model.')
    regr.fit(iris_X, iris_y)

    # Format scores to be written to metadata.
    # Beside cv score, we can have other scores (e.g. hold-out)
    scores = {
        'r2': {'cv': r2_cv_score},
        'mean_squared_error': {'cv': mse_cv_score},
    }

    # Save the model
    model_location = os.path.join(s.MODEL_DIR, '{}.p'.format(model_filename))
    logger.info('Saving serialized model: {}'.format(model_filename))
    joblib.dump(regr, model_location)
    model_identifier = hashlib.sha1(
        str(get_git_commit()).encode('utf-8') +
        str(datetime.datetime.now()).encode('utf-8'))\
        .hexdigest()

    # Save relevant metadata.
    logger.info('Saving model metadata: {}'.format(model_filename))
    model_description = 'Predicting petal length (regression)'
    testing_strategy = '5-fold cross validation, using mean ' \
                       'to aggregate fold metrics, no hold-out set.'
    feature_names = list(iris_X.columns.values)
    metadata_to_file(path=s.MODEL_METADATA_DIR,
                     filename='{}-{}'.format(model_filename, model_identifier),
                     metadata={
                        'model_location': model_location,
                        'model_type': str(regr.__class__),
                        'model_description': model_description,
                        'model_identifier': model_identifier,
                        'sklearn_object': regr,
                        'input_data_location': data_location,
                        'input_data_identifier': None,
                        'feature_names': feature_names,
                        'testing_strategy': testing_strategy,
                        'scores': scores},
                     logger=logger)


if __name__ == '__main__':
    main()
