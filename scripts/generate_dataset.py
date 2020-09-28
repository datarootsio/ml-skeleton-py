#!/usr/bin/env python3
from ml_skeleton_py import etl
import argparse


def generate() -> None:
    """
    Load the dataset, remove outliers and store in data directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="creditcard.csv", help="raw dataset to generate train and test data")
    args = parser.parse_args()
    etl.generate(args.dataset)


if __name__ == "__main__":
    generate()
