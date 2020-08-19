#!/usr/bin/env python3
import click

from ml_skeleton_py import model


@click.command()
@click.option("--model_name", default="lr")
@click.option("--dataset", default="creditcard.csv")
def train(model_name: str, dataset: str) -> None:
    """
    Train a model on a dataset and store the model and its results.

    Parameters:
        model_name (str): the model_name that you want to use as a save
                     default:
                        "lr": logistic regression

        dataset (str): the dataset on which you want to train

    Returns:
        None
    """
    model.train(model_name, dataset)


if __name__ == "__main__":
    train()
