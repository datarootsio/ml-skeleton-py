#!/usr/bin/env python3

import os
import sys

import click

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import model


@click.command()
@click.option("--model-filename", default="model")
@click.option("--input-data-filename", default="iris.csv")
def train(model_filename, input_data_filename):
    # model.train(model_filename, input_data_filename)
    model.train()


if __name__ == "__main__":
    train()
