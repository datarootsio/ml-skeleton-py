#!/usr/bin/env python3

import os
import sys

import click

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import etl


@click.command()
@click.option("--dataset", type=str, default="creditcard.csv")
def generate(dataset: str) -> None:
    """
    Load the dataset, remove the outliers and store in transformed data directory.

    Parameters:
        dataset (str): the dataset that you want to preprocess and transform

    Returns:
        None
    """
    etl.generate(dataset)


if __name__ == "__main__":
    generate()