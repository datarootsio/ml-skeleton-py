#!/usr/bin/env python3
import click

from ml_skeleton_py import etl


@click.command()
@click.option("--dataset", type=str, default="creditcard.csv")
def generate(dataset: str) -> None:
    """
    Load the dataset, remove outliers and store in data directory.

    Parameters:
        dataset (str): the dataset that you want to preprocess and transform

    Returns:
        None
    """
    etl.generate(dataset)
    etl.generate_test(dataset)


if __name__ == "__main__":
    generate()
