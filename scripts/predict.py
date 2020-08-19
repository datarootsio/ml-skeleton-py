#!/usr/bin/env python3
import click

from ml_skeleton_py import model


@click.command()
@click.option("--model_name", default="lr.p")
@click.option("--input_df", default="test_balanced_creditcard.csv")
@click.option("--output_df", default="predictions.csv")
def predict(model_name: str, input_df: str, output_df: str) -> None:
    """Predict new values using a serialized model.
    Parameters:
        model_name (str): name find the model to load (including extension)

        input_df (str): the input features to use to generate prediction on

        output_df (str): the output data file to store predictions as

    Returns:
        None
    """
    model.predict_from_file(model_name, input_df, output_df)


if __name__ == "__main__":
    predict()
