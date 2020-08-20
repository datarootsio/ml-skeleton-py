"""
EXAMPLE of training procedure on a highly imbalanced credit fraud dataset.

The dataset is retrieved from:
https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
import logging
from ml_skeleton_py import settings
from ml_skeleton_py.helper import save_pickle

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def train(model_name: str, dataset: str) -> None:
    """
    Train models using X_train and y_train with a specific classifier.

    Trains a specific classifier with a set of optimized hyperparameters
    in a 5fold-CV. The training results with the accompanying model is
    saved in ./models/

    Parameters:
        model_name (str): the model_name that you want to use as a save
                     default:
                        "lr": logistic regression

        dataset (str): the dataset on which you want to train

    Returns:
        None

    """
    # loading data
    df = pd.read_csv(os.path.join(settings.DATA_TRANSFORMED, dataset))
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # preprocessing
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    rus = RandomUnderSampler(replacement=False)
    X, y = rus.fit_resample(X, y)

    # In this specific example logistic regression was chosen as
    # the most optimal model after running several experiments.
    classifier = LogisticRegression(max_iter=4000, penalty="l2", C=0.01)

    # training
    classifier.fit(X, y)
    training_score = cross_val_score(classifier, X, y, cv=5, scoring="roc_auc")
    logger.info(f"Classifier: {classifier.__class__.__name__}")
    logger.info(
        "Has a training score "
        + f"of {round(training_score.mean(), 2) * 100} % roc_auc"
    )

    # saving
    predict_pipeline = make_pipeline(scaler, classifier)
    pred_result = {
        "clf": model_name,
        "training score roc_auc": training_score.mean(),
        "model": predict_pipeline,
    }
    model_path = os.path.join(settings.MODEL_DIR, model_name) + ".p"
    save_pickle(pred_result, model_path)
