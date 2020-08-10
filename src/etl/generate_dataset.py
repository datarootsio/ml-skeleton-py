# EXAMPLE adapted from
# https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import logging
from .. import settings as s
import requests

logger = logging.getLogger(__name__)


def generate():
    # load dataset
    if not os.path.isfile(os.path.join(s.DATA_RAW, "creditcard.csv")):
        print("creditcard.csv not found in " + os.path.join(s.DATA_RAW))
        print("please download the file from url = 'https://www.kaggle.com/mlg-ulb/creditcardfraud/download' and place it in " + s.DATA_RAW)
        return 
        # open(os.path.join(s.DATA_RAW, "creditcard.zip"), 'wb').write(r.content)


    df = pd.read_csv(os.path.join(s.DATA_RAW, "creditcard.csv"))

    logger.info("Preprocessing dataset from raw to tranformed")
    logger.info("Scaling data")
    # preprocess data
    # Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

    rob_scaler_amount, rob_scaler_time = RobustScaler(), RobustScaler()
    rob_scaler_amount.fit(df["Amount"].values.reshape(-1, 1))
    rob_scaler_time.fit(df["Time"].values.reshape(-1, 1))

    def scaler(df, rob_scaler_amount, rob_scaler_time):
        df["scaled_amount"] = rob_scaler_amount.transform(
            df["Amount"].values.reshape(-1, 1)
        )
        df["scaled_time"] = rob_scaler_time.transform(df["Time"].values.reshape(-1, 1))
        df.drop(["Time", "Amount"], axis=1, inplace=True)

        scaled_amount = df["scaled_amount"]
        scaled_time = df["scaled_time"]

        df.drop(["scaled_amount", "scaled_time"], axis=1, inplace=True)
        df.insert(0, "scaled_amount", scaled_amount)
        df.insert(1, "scaled_time", scaled_time)
        return df

    df = scaler(df, rob_scaler_amount, rob_scaler_time)

    # split data original df

    print(
        "No Frauds",
        round(df["Class"].value_counts()[0] / len(df) * 100, 2),
        "% of the dataset",
    )
    print(
        "Frauds",
        round(df["Class"].value_counts()[1] / len(df) * 100, 2),
        "% of the dataset",
    )
    print("")

    # Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

    # Lets shuffle the data before creating the subsamples
    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df["Class"] == 1]
    non_fraud_df = df.loc[df["Class"] == 0][:492]
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    new_df = normal_distributed_df.sample(frac=1, random_state=42)

    ## outlier removal
    logger.info("Outlier removal")
    # # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
    v14_fraud = new_df["V14"].loc[new_df["Class"] == 1].values
    q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
    v14_iqr = q75 - q25
    v14_cut_off = v14_iqr * 1.5
    V14_LOWER, V14_UPPER = q25 - v14_cut_off, q75 + v14_cut_off
    outliers = [x for x in v14_fraud if x < V14_LOWER or x > V14_UPPER]
    print("Feature V14 Outliers for Fraud Cases: {}".format(len(outliers)))

    # -----> V12 removing outliers from fraud transactions
    v12_fraud = new_df["V12"].loc[new_df["Class"] == 1].values
    q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
    v12_iqr = q75 - q25
    v12_cut_off = v12_iqr * 1.5
    V12_LOWER, V12_UPPER = q25 - v12_cut_off, q75 + v12_cut_off
    outliers = [x for x in v12_fraud if x < V12_LOWER or x > V12_UPPER]
    print("Feature V12 Outliers for Fraud Cases: {}".format(len(outliers)))

    # Removing outliers V10 Feature
    v10_fraud = new_df["V10"].loc[new_df["Class"] == 1].values
    q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
    v10_iqr = q75 - q25
    v10_cut_off = v10_iqr * 1.5
    V10_LOWER, V10_UPPER = q25 - v10_cut_off, q75 + v10_cut_off
    outliers = [x for x in v10_fraud if x < V10_LOWER or x > V10_UPPER]
    print("Feature V10 Outliers for Fraud Cases: {}".format(len(outliers)))

    outlier_params = {
        "V14_UPPER": V14_UPPER,
        "V14_LOWER": V14_LOWER,
        "V12_UPPER": V12_UPPER,
        "V12_LOWER": V12_LOWER,
        "V10_UPPER": V10_UPPER,
        "V10_LOWER": V10_LOWER,
    }

    def remove_outliers(new_df, params):
        """
        :removes outliers:
        """
        new_df = new_df.drop(
            new_df[
                (new_df["V14"] > params["V14_UPPER"])
                | (new_df["V14"] < params["V14_LOWER"])
            ].index
        )
        new_df = new_df.drop(
            new_df[
                (new_df["V12"] > params["V12_UPPER"])
                | (new_df["V12"] < params["V12_LOWER"])
            ].index
        )
        new_df = new_df.drop(
            new_df[
                (new_df["V10"] > params["V10_UPPER"])
                | (new_df["V10"] < params["V10_LOWER"])
            ].index
        )
        print("Number of Instances after outliers removal: {}".format(len(new_df)))
        return new_df

    new_df = remove_outliers(new_df, outlier_params)

    preprocessing_functions = {
        "scale_amount": rob_scaler_amount,
        "scale_time": rob_scaler_time,
    }
    pickle.dump(
        preprocessing_functions, open(os.path.join(s.MODEL_DIR, "preprocessor.p"), "wb")
    )

    # New_df is from the random undersample data (fewer instances)
    X = new_df.drop("Class", axis=1)
    y = new_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Turn the values into an array for feeding the classification algorithms.
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    pickle.dump(X, open(os.path.join(s.DATA_TRANSFORMED, "X_train.p"), "wb"))
    pickle.dump(y, open(os.path.join(s.DATA_TRANSFORMED, "y_train.p"), "wb"))
    pickle.dump(X, open(os.path.join(s.DATA_TRANSFORMED, "X_test.p"), "wb"))
    pickle.dump(y, open(os.path.join(s.DATA_TRANSFORMED, "y_test.p"), "wb"))

    logger.info("Done")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    generate()
