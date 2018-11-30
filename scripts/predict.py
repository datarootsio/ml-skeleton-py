#!/usr/bin/env python3

import os
import sys

import click

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import model

@click.command()
@click.option('--model-name', default='model')
@click.option('--input-df', default='model')
@click.option('--output-df', default='predictions.csv')
def predict(model_name, input_df, output_df):
    model.predict_from_file(model_name, input_df, output_df)


if __name__ == '__main__':
    predict()