"""
EXAMPLE adapted from kaggle.

This script loads the data, removes the outliers and saves the dataframe.

See:
https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
"""

import logging

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import LocalOutlierFactor

from ml_skeleton_py import settings as s

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def generate(raw_data_loc: str, transformed_data_loc: str) -> None:
    """
    Load data, remove outliers and write transformed data.

    Parameters:
        raw_data_loc (str): the location of the raw data you want to load

        transformed_data_loc (str): the location of the generated transformed
            data you want to save
    """
    logger.info(f"Loading dataset from: {raw_data_loc}")

    df = pd.read_csv(raw_data_loc)

    # Remove outliers
    df = remove_outliers(df)

    # Give some overview about the generated dataset
    logger.info("Preprocessing dataset from raw to transformed")
    no_frauds = round(df[s.TARGET_VARIABLE].value_counts()[0] / len(df) * 100, 2)
    frauds = round(df[s.TARGET_VARIABLE].value_counts()[1] / len(df) * 100, 2)
    logger.info(f"No Frauds {no_frauds} % of the dataset")
    logger.info(f"Frauds {frauds} % of the dataset")

    # save data frame into disk
    df.to_csv(transformed_data_loc, index=False)
    logger.info("Training data has been saved into disk!")


def remove_outliers(df: pd.DataFrame, **kwargs: int) -> pd.DataFrame:
    """
    Remove outliers using local outlier factor algorithm.

    Parameters:
        df (pd.DataFrame): data frame with outliers

    Returns:
        df_outlier_removed (pd.DataFrame): data frame without outliers
    """
    df_outlier_removed = df.copy(deep=True)
    n_rows = df_outlier_removed.shape[0]

    # Fit a basic local outlier factor to detect outliers
    lof = LocalOutlierFactor(**kwargs)
    df_outlier_removed["is_outlier"] = lof.fit_predict(
        df_outlier_removed[["V10", "V12", "V14"]]
    )

    df_outlier_removed = df_outlier_removed[
        df_outlier_removed.is_outlier != -1
    ]  # -1 represents outliers

    # Report number of removed rows
    n_filtered_rows = df_outlier_removed.shape[0]
    logger.info(
        "{} outliers are filtered out of {} rows.".format(
            n_rows - n_filtered_rows, n_filtered_rows
        )
    )

    # Remove temporary is_outlier column
    df_outlier_removed = df_outlier_removed.drop("is_outlier", axis=1)
    return df_outlier_removed
