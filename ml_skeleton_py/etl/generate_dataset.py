"""
EXAMPLE adapted from kaggle.

This script loads the data, removes the outliers and saves the dataframe.

See:
https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
"""

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import logging
from ml_skeleton_py import settings as s
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def generate(dataset: str) -> None:
    """
    Load data, remove outliers and write train and test data.

    Parameters:
        dataset: the filename of the dataset that you want to load
    """
    logger.info(f"Loading dataset {dataset}")
    if not os.path.isfile(os.path.join(s.DATA_RAW, dataset)):
        logger.error("creditcard.csv not found in {}"
                     "please download the file from url"
                     "https://www.kaggle.com/mlg-ulb/"
                     "creditcardfraud/download".format(s.DATA_RAW))

    df = pd.read_csv(os.path.join(s.DATA_RAW, dataset))

    # Remove outliers
    df = remove_outliers(df)

    # Give some overview about the generated dataset
    logger.info("Preprocessing dataset from raw to tranformed")
    target_col = "Class"
    no_frauds = round(df[target_col].value_counts()[0] / len(df) * 100, 2)
    frauds = round(df[target_col].value_counts()[1] / len(df) * 100, 2)
    logger.info(f"No Frauds {no_frauds} % of the dataset")
    logger.info(f"Frauds {frauds} % of the dataset")

    # save data frame into disk
    df.to_csv(os.path.join(s.DATA_TRANSFORMED, dataset), index=False)
    logger.info("Training data has been saved into disk!")


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers using local outlier factor algorithm

    Parameters:
        df: data frame with outliers

    Returns:
        df: data frame without outliers
    """
    n_rows = df.shape[0]

    # Fit a basic local outlier factor to detect outliers
    lof = LocalOutlierFactor()
    df['is_outlier'] = lof.fit_predict(df[["V10", "V12", "V14"]])
    df = df[df.is_outlier != -1]  # -1 represents outliers

    # Report number of removed rows
    n_filtered_rows = df.shape[0]
    logger.info("{} outliers are filtered out of {} rows."
                .format(n_rows - n_filtered_rows, n_filtered_rows)
                )

    # Remove temporary is_outlier column
    df = df.drop('is_outlier', axis=1)
    return df
