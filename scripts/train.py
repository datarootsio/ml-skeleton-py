#!/usr/bin/env python3
import argparse
from ml_skeleton_py import model


def train() -> None:
    """
    Train a model on a dataset and store the model and its results.

    Parameters:
        dataset (str): the dataset on which you want to train

        model_name (str): the model_name that you want to use as a save
                     default:
                        "lr": logistic regression

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="creditcard.csv", help="raw dataset to generate train and test data")
    parser.add_argument("--model-name", default="lr", help="the serialized model name default lr referring to "
                                                           "logistic regression")
    args = parser.parse_args()
    model.train(args.dataset, args.model_name)


if __name__ == "__main__":
    train()
