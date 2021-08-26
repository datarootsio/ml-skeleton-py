"""
EXAMPLE of training procedure on a highly imbalanced credit fraud dataset.

The dataset is retrieved from:
https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets.
"""
import logging
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from ml_skeleton_py import settings as s

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def train(dataset_loc: str, model_dir: str, model_name: str = "lr") -> None:
    """
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
    y_train = df.pop(s.TARGET_VARIABLE)
    X_train = df

    # pre-processing
    scaler = RobustScaler()

    # In this specific example logistic regression was chosen as
    # the most optimal models after running several experiments.
    classifier = LogisticRegression(max_iter=4000, penalty="l2", C=0.01)

    # create pipeline
    pipeline = make_pipeline(scaler, classifier)

    # training
    pipeline.fit(X_train, y_train)
    training_score = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="roc_auc"
    )

    auc_roc = round(training_score.mean(), 2)
    logger.info(f"Classifier: {pipeline.__class__.__name__}")
    logger.info("Has a training score " + f"of {auc_roc} roc_auc")
    check_performance(auc_roc)
    # Serialize and dump trained pipeline to disk
    pred_result = {
        "model_name": model_name,
        "roc_auc": training_score.mean(),
        "deserialized_model": pipeline,
    }

    model_location = os.path.join(model_dir, model_name) + ".joblib"
    with open(model_location, "wb") as f:
        # Serialize pipeline and compress it with the max factor 9
        joblib.dump(pred_result, f, compress=9)


def check_performance(auc_roc: float) -> None:
    if auc_roc < s.EXPECTED_MIN_AUC:
        raise Exception(
            "The auc roc is less than the expected, "
            "please check your data manipulation or "
            "training parameters!"
        )
    else:
        # Performance is more than the expected
        pass
