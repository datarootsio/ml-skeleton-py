"""
EXAMPLE adapted from kaggle.

This script loads the data, removes the outliers and saves the dataframe.

See: https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import logging
from ml_skeleton_py import settings

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
    df = df.drop(
        df[(df["V14"] > params["V14_upper"]) | (df["V14"] < params["V14_lower"])].index
    )
    df = df.drop(
        df[(df["V12"] > params["V12_upper"]) | (df["V12"] < params["V12_lower"])].index
    )
    df = df.drop(
        df[(df["V10"] > params["V10_upper"]) | (df["V10"] < params["V10_lower"])].index
    )
    logger.info(f"Number of Instances after outliers removal: {len(df)}")
    return df


def generate(dataset: str) -> pd.DataFrame:
    """
    Load the data, removes outliers, subsamples the data and returns the traing and test datasets.

    Parameters:
        dataset (str): the input dataset

    Returns:
        X_train, y_train, X_test, y_test (tuple): the training and test datasets with labels
    """
    logger.info(f"Loading dataset {dataset}")
    if not os.path.isfile(os.path.join(settings.DATA_RAW, dataset)):
        logger.info("creditcard.csv not found in " + os.path.join(settings.DATA_RAW))
        logger.info(
            f"please download the file from url = \
            'https://www.kaggle.com/mlg-ulb/creditcardfraud/download' \
            and place it in {settings.DATA_RAW}"
        )
        return

    df = pd.read_csv(os.path.join(settings.DATA_RAW, dataset))

    logger.info("Preprocessing dataset from raw to tranformed")
    no_frauds = round(df["Class"].value_counts()[0] / len(df) * 100, 2)
    frauds = round(df["Class"].value_counts()[1] / len(df) * 100, 2)
    logger.info(f"No Frauds {no_frauds} % of the dataset")
    logger.info(f"Frauds {frauds} % of the dataset")

    # Removing outliers
    logger.info("Outlier removal")
    # # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
    v14_fraud = df["V14"].loc[df["Class"] == 1].values
    q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
    v14_iqr = q75 - q25
    v14_cut_off = v14_iqr * 1.5
    V14_lower, V14_upper = q25 - v14_cut_off, q75 + v14_cut_off
    outliers = [x for x in v14_fraud if x < V14_lower or x > V14_upper]
    logger.info(f"Feature V14 Outliers for Fraud Cases: {len(outliers)}")

    # -----> V12 removing outliers from fraud transactions
    v12_fraud = df["V12"].loc[df["Class"] == 1].values
    q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
    v12_iqr = q75 - q25
    v12_cut_off = v12_iqr * 1.5
    V12_lower, V12_upper = q25 - v12_cut_off, q75 + v12_cut_off
    outliers = [x for x in v12_fraud if x < V12_lower or x > V12_upper]
    logger.info(f"Feature V12 Outliers for Fraud Cases: {len(outliers)}")

    # -----> Removing outliers V10 Feature
    v10_fraud = df["V10"].loc[df["Class"] == 1].values
    q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
    v10_iqr = q75 - q25
    v10_cut_off = v10_iqr * 1.5
    V10_lower, V10_upper = q25 - v10_cut_off, q75 + v10_cut_off
    outliers = [x for x in v10_fraud if x < V10_lower or x > V10_upper]
    logger.info(f"Feature V10 Outliers for Fraud Cases: {len(outliers)}")

    outlier_params = {
        "V14_upper": V14_upper,
        "V14_lower": V14_lower,
        "V12_upper": V12_upper,
        "V12_lower": V12_lower,
        "V10_upper": V10_upper,
        "V10_lower": V10_lower,
    }

    df = remove_outliers(df, outlier_params)

    # save dataframe with removed outliers
    df.to_csv(os.path.join(settings.DATA_TRANSFORMED, dataset), index=0)

    logger.info("Done!")
    return df
