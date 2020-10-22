import argparse
import os

from ml_skeleton_py import model
from ml_skeleton_py import settings as s


def train() -> None:
    """
    Train a model on a dataset and store the model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="creditcard.csv",
        help="raw dataset to generate train and test data",
    )
    parser.add_argument(
        "--model-name",
        default="lr",
        help="the serialized model name default lr " "referring to logistic regression",
    )
    args = parser.parse_args()
    transformed_data_dir = os.path.join(s.DATA_TRANSFORMED, args.dataset)
    model.train(transformed_data_dir, s.MODEL_DIR, args.model_name)


if __name__ == "__main__":
    train()
