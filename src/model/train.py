"""Train example. Regression on iris dataset."""

import os
import logging

from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import pandas as pd

from .. import settings as s
from sklearn.model_selection import KFold, cross_validate

from mlmonkey.metadata import ModelMetadata

logger = logging.getLogger(__name__)


def train(model_filename, input_data_filename):
    """Train and save a model. Calculate evaluation metrics. Write metadata.

    :param model_filename: name of file that stores serialized model
    (without extension)
    :param input_data_filename: name of file that holds data used to train
    the model (with extension)
    """
    # Load the iris dataset
    logging.info('Loading iris dataset')
    data_location = os.path.join(s.DATA_TRANSFORMED, input_data_filename)
    iris = pd.read_csv(data_location)
    iris = iris.sample(frac=1)  # shuffle
    # Prepare train and target columns
    iris_X = iris.drop(columns=['species', 'petal_length'])
    iris_y = iris['petal_length']

    # Create linear regression object
    regr_model = RandomForestRegressor()

    # Calculating CV score(s)
    logger.info('Performing cross validation')
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(regr_model, iris_X, iris_y, cv=cv,
                            scoring=['r2', 'neg_mean_squared_error'],
                            return_train_score=False, verbose=1)
    r2_cv_score = scores['test_r2'].mean()
    # cross_validate outputs negative mse
    mse_cv_score = - scores['test_neg_mean_squared_error'].mean()

    # Calculate predictions, to plot actual-vs-predicted later.
    predictions = cross_val_predict(regr_model, iris_X, iris_y, cv=5)

    # Train a model using the whole dataset
    logger.info('Fitting linear model.')
    regr_model.fit(iris_X, iris_y)

    # Create metadata
    model_description = 'Predicting petal length (regression)'
    model_location = os.path.join(s.MODEL_DIR, '{}.joblib'.format(model_filename))
    feature_names = iris_X.columns.values.tolist()

    testing_strategy = '5-fold cross validation, using mean ' \
                       'to aggregate fold metrics, no hold-out set.'
    extra_metadata = {
        'data_type': 'csv',
        'actual_and_predicted': {
            'actual': list(iris_y),
            'predicted': list(predictions)
        }
    }
    metadata = ModelMetadata(model_location, regr_model, model_description,
                             data_location, None, feature_names,
                             list(regr_model.feature_importances_), testing_strategy, None,
                             extra_metadata=extra_metadata)

    # add scores to metadata
    metadata.add_score('r2', 'cv', r2_cv_score)
    metadata.add_score('mean_squared_error', 'cv', mse_cv_score)

    # Save the model.
    logger.info('Saving serialized model: {}'.format(model_filename))
    joblib.dump(regr_model, model_location)

    # Save metadata to file
    metadata.save_to_file(s.MODEL_METADATA_DIR)
