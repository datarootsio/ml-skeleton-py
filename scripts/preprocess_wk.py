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

RAW_DATA_DIR = "../data/raw/"
TRANSFORMED_DATA_DIR = "../data/transformed/"


def generate_dataset():
    # load dataset
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, "creditcard.csv"))

    # preprocess data
    # Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df["scaled_amount"] = rob_scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = rob_scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    scaled_amount = df["scaled_amount"]
    scaled_time = df["scaled_time"]

    df.drop(["scaled_amount", "scaled_time"], axis=1, inplace=True)
    df.insert(0, "scaled_amount", scaled_amount)
    df.insert(1, "scaled_time", scaled_time)

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

    X = df.drop("Class", axis=1)
    y = df["Class"]

    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in sss.split(X, y):
        # print("Train:", train_index, "Test:", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    # We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
    # original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the Distribution of the labels

    # Turn into an array
    original_Xtrain = original_Xtrain.values
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values

    # See if both the train and test label distribution are similarly distributed
    train_unique_label, train_counts_label = np.unique(
        original_ytrain, return_counts=True
    )
    test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
    # print('-' * 100)

    print("Label Distributions: ")
    print(train_counts_label / len(original_ytrain))
    print(test_counts_label / len(original_ytest))
    print("")

    # Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

    # Lets shuffle the data before creating the subsamples

    df = df.sample(frac=1)

    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df["Class"] == 1]
    non_fraud_df = df.loc[df["Class"] == 0][:492]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # Shuffle dataframe rows
    new_df = normal_distributed_df.sample(frac=1, random_state=42)

    ## outlier removal
    # # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
    v14_fraud = new_df["V14"].loc[new_df["Class"] == 1].values
    q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
    # print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
    v14_iqr = q75 - q25
    # print('iqr: {}'.format(v14_iqr))

    v14_cut_off = v14_iqr * 1.5
    v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
    # print('Cut Off: {}'.format(v14_cut_off))
    # print('V14 Lower: {}'.format(v14_lower))
    # print('V14 Upper: {}'.format(v14_upper))

    outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
    print("Feature V14 Outliers for Fraud Cases: {}".format(len(outliers)))
    # print('V10 outliers:{}'.format(outliers))

    new_df = new_df.drop(
        new_df[(new_df["V14"] > v14_upper) | (new_df["V14"] < v14_lower)].index
    )
    print("Number of Instances after outliers removal: {}".format(len(new_df)))
    # print('----' * 44)

    # -----> V12 removing outliers from fraud transactions
    v12_fraud = new_df["V12"].loc[new_df["Class"] == 1].values
    q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
    v12_iqr = q75 - q25

    v12_cut_off = v12_iqr * 1.5
    v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
    # print('V12 Lower: {}'.format(v12_lower))
    # print('V12 Upper: {}'.format(v12_upper))
    outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
    # print('V12 outliers: {}'.format(outliers))
    print("Feature V12 Outliers for Fraud Cases: {}".format(len(outliers)))
    new_df = new_df.drop(
        new_df[(new_df["V12"] > v12_upper) | (new_df["V12"] < v12_lower)].index
    )
    print("Number of Instances after outliers removal: {}".format(len(new_df)))
    # print('----' * 44)

    # Removing outliers V10 Feature
    v10_fraud = new_df["V10"].loc[new_df["Class"] == 1].values
    q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
    v10_iqr = q75 - q25

    v10_cut_off = v10_iqr * 1.5
    v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
    # print('V10 Lower: {}'.format(v10_lower))
    # print('V10 Upper: {}'.format(v10_upper))
    outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
    # print('V10 outliers: {}'.format(outliers))
    print("Feature V10 Outliers for Fraud Cases: {}".format(len(outliers)))
    new_df = new_df.drop(
        new_df[(new_df["V10"] > v10_upper) | (new_df["V10"] < v10_lower)].index
    )
    print("Number of Instances after outliers removal: {}".format(len(new_df)))

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

    pickle.dump(X, open(os.path.join(TRANSFORMED_DATA_DIR, "X_train.p"), "wb"))
    pickle.dump(y, open(os.path.join(TRANSFORMED_DATA_DIR, "y_train.p"), "wb"))
    pickle.dump(X, open(os.path.join(TRANSFORMED_DATA_DIR, "X_test.p"), "wb"))
    pickle.dump(y, open(os.path.join(TRANSFORMED_DATA_DIR, "y_test.p"), "wb"))
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    generate_dataset()
