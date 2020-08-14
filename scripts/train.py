#!/usr/bin/env python3

import os
import sys

import click

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import model


@click.command()
@click.option("--model_name", default="lr")
@click.option("--dataset", default="creditcard.csv")
def train(model_name: str, dataset: str) -> None:
    """
    Train a model on a dataset and store the model and its results.

    Parameters:
        model_name (str): the model that you want to train
                     options:
                        "lr": logistic regression
                        "knn": k nearest neighbors
                        "svc": support vector classifier
                        "dt": decision tree

        dataset (str): the dataset on which you want to train

    Returns:
        None
    """
    model.train(model_name, dataset)


if __name__ == "__main__":
	model_name = "lr"
	dataset = "creditcard.csv"
    train(model_name, dataset)
