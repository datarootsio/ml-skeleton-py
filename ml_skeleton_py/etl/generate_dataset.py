"""
EXAMPLE adapted from kaggle.

This script loads the data, removes the outliers and saves the dataframe.

See:
https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import logging
from ml_skeleton_py import settings as s
from typing import Optional
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def remove_outliers(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Remove outliers depending on the cut off thresholds given in params.

    Parameters:
        df (pd.DataFrame): dataframe with outliers to be removed

        params (dict): dictionary with cut-off thresholds

    Return:
        df (pd.DataFrame): dataframe with removed outliers
    """
    df_dropped = df.copy(deep=True)
    for variable in ["V10", "V12", "V14"]:
        upper_outliers = df_dropped[variable] > params[f"{variable}_upper"]
        lower_outliers = df_dropped[variable] < params[f"{variable}_lower"]
        filter_outliers = upper_outliers | lower_outliers
        df_dropped = df_dropped.drop(df_dropped[filter_outliers].index)
    message = f"Number of Instances after outliers removal: {len(df_dropped)}"
    logger.info(message)
    return df_dropped


def generate(dataset: str) -> Optional[pd.DataFrame]:
    """
    Load data, remove outliers and return the traing and test sets.

    Parameters:
        dataset (str): the filename of the dataset that you want to load

    Returns:
        df Optional(pd.DataFrame): preprocessed dataframe
    """
    logger.info(f"Loading dataset {dataset}")
    if not os.path.isfile(os.path.join(s.DATA_RAW, dataset)):
        logger.info("creditcard.csv not found in " + s.DATA_RAW)
        logger.info(
            f"please download the file from url = \
            'https://www.kaggle.com/mlg-ulb/creditcardfraud/download' \
            and place it in {s.DATA_RAW}"
        )
        return

    df = pd.read_csv(os.path.join(s.DATA_RAW, dataset))

    logger.info("Preprocessing dataset from raw to tranformed")
    no_frauds = round(df["Class"].value_counts()[0] / len(df) * 100, 2)
    frauds = round(df["Class"].value_counts()[1] / len(df) * 100, 2)
    logger.info(f"No Frauds {no_frauds} % of the dataset")
    logger.info(f"Frauds {frauds} % of the dataset")

    # Removing outliers
    logger.info("Outlier removal")
    # --> V10,V12,14 Removing Outliers
    # (Highest Negative Correlated with Labels)
    outlier_params = {}
    for variable in ["V10", "V12", "V14"]:
        fraud = df[variable].loc[df["Class"] == 1].values
        q25, q75 = np.percentile(fraud, 25), np.percentile(fraud, 75)
        iqr = q75 - q25
        cut_off = iqr * 1.5

        lower_cutoff = q25 - cut_off
        upper_cutoff = q75 + cut_off
        outlier_params[f"{variable}_lower"] = lower_cutoff
        outlier_params[f"{variable}_upper"] = upper_cutoff

        outliers = [x for x in fraud if x < lower_cutoff or x > upper_cutoff]
        logger.info(
            f"Feature {variable} Outliers for Fraud Cases:\
         {len(outliers)}"
        )

    df = remove_outliers(df, outlier_params)

    # save dataframe with removed outliers
    df.to_csv(os.path.join(s.DATA_TRANSFORMED, dataset), index=0)

    logger.info("Done!")
    return df


def generate_test(dataset: str) -> None:
    """
    Load transformed data, create and saves a balanced and imbalanced test set.

    Parameters:
        dataset (str): the filename of the dataset that you want to load

    Returns:
        None
    """

    logger.info("Creating an imbalanced and balanced sample test set")
    # Load Data
    df = pd.read_csv(os.path.join(s.DATA_TRANSFORMED, dataset))

    # Create X
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Create imbalanced test set X_test
    X_test = X.sample(frac=0.01)

    # save imbalanced dataframe
    file_path = os.path.join(s.DATA_TRANSFORMED, "test_imbalanced_" + dataset)
    X_test.to_csv(file_path, index=0)

    # Create balanced test set X_test
    rus = RandomUnderSampler(replacement=False)
    X, y = rus.fit_resample(X, y)
    X_test = X.sample(frac=0.1)

    # save dataframe with removed outliers
    file_path = os.path.join(s.DATA_TRANSFORMED, "test_balanced_" + dataset)
    X_test.to_csv(file_path, index=0)
    logger.info("Done!")
