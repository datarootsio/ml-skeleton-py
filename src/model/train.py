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
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import r2_score, mean_squared_error

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

    # Split the data into training/testing sets
    iris_X_train, iris_X_test, iris_y_train, iris_y_test = \
        train_test_split(iris_X, iris_y, test_size=0.2, random_state=0)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Calculating CV score(s)
    logger.info('Fitting linear model')
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(regr, iris_X_train, iris_y_train, cv=cv,
                            scoring=['r2', 'neg_mean_squared_error'],
                            return_train_score=False, verbose=1)
    r2_cv_score = scores['test_r2'].mean()
    # cross_validate outputs negative mse
    mse_cv_score = - scores['test_neg_mean_squared_error'].mean()

    # Train a model using the whole training set
    logger.info('Fitting linear model')
    regr.fit(iris_X_train, iris_y_train)

    # Calculate test scores.
    r2_test_score = r2_score(regr.predict(iris_X_test), iris_y_test)
    mse_test_score = mean_squared_error(regr.predict(iris_X_test), iris_y_test)

    # Format scores to be written to metadata.
    scores = {
        'r2': {'cv': r2_cv_score, 'test': r2_test_score},
        'mse': {'cv': mse_cv_score, 'test': mse_test_score}
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
    logger.info('Saving model metadata for model: {}'.format(model_filename))
    model_description = 'Predicting petal length (regression)'
    testing_strategy = '5-fold cross validation, 20% testing (hold-out) set.'
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
                        'testing_strategy': testing_strategy,
                        'scores': scores},
                     logger=logger)


if __name__ == '__main__':
    main()
