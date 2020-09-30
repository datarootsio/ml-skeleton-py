"""
EXAMPLE of training procedure on a highly imbalanced credit fraud dataset.

The dataset is retrieved from:
https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets.
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd
import sklearn.pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def train(dataset_loc: str, model_dir: str, model_name: str = "lr") -> None:
    """
    Train models using X_train and y_train with a specific classifier.

    Trains a specific classifier with a set of optimized hyper-parameters
    in a 5 fold-CV. The training results with the accompanying models is
    saved in ./models/

    Parameters:
        dataset_loc (str): the dataset path on which we want to train

        model_dir (str): directory of the serialized ml models

        model_name (str): the model_name that we want to use as a save
             default:
                "lr": logistic regression
    Returns:
        None

    """
    # loading data
    df = pd.read_csv(dataset_loc)

    # Separate X and y
    y = df.pop("Class")
    X = df

    # pre-processing
    scaler = RobustScaler()

    # In this specific example logistic regression was chosen as
    # the most optimal models after running several experiments.
    classifier = LogisticRegression(max_iter=4000, penalty="l2", C=0.01)

    # create pipeline
    pipeline = make_pipeline(scaler, classifier)

    # training
    pipeline.fit(X, y)
    training_score = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    logger.info(f"Classifier: {pipeline.__class__.__name__}")
    logger.info(
        "Has a training score "
        + f"of {round(training_score.mean(), 2) * 100} % roc_auc"
    )

    dump_model(pipeline, model_name, training_score, model_dir)


def dump_model(
        pipeline: sklearn.pipeline,
        model_name: str,
        training_score: np.ndarray,
        model_dir: str,
) -> None:
    """
    Dump serialized trained pipeline to disk

    Parameters:
        pipeline (sklearn.pipeline): Fitted pipeline object
                                     that we want to serialize

        model_name (str): Name of the models that we want to serialize
                     default:
                        "lr": logistic regression
        training_score (np.ndarray): ROC AUC scores of each CV

        model_dir (str): directory of the serialized ml models

    Returns:
        None
    """

    pred_result = {
        "model_name": model_name,
        "roc_auc": training_score.mean(),
        "deserialized_model": pipeline,
    }
    model_location = os.path.join(model_dir, model_name) + ".joblib"
    with open(model_location, "wb") as f:
        # Serialize pipeline and compress it with the max factor 9
        joblib.dump(pred_result, f, compress=9)
