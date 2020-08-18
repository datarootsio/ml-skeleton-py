#!/usr/bin/env python3
import click

from ml_skeleton_py import model


@click.command()
@click.option("--model_name", default="model")
@click.option("--input_df", default="X_test.p")
@click.option("--output_df", default="predictions.csv")
def predict(model_name, input_df, output_df):
    model.predict_from_file(model_name, input_df, output_df)


if __name__ == "__main__":
    predict()
