import argparse
import logging
import os

from ml_skeleton_py import etl
from ml_skeleton_py import settings as s

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def generate() -> None:
    """
    Load the dataset, remove outliers and store in data directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="creditcard.csv",
        help="raw dataset to generate train and test data",
    )
    args = parser.parse_args()

    input_location = os.path.join(s.DATA_RAW, args.dataset)
    output_location = os.path.join(s.DATA_TRANSFORMED, args.dataset)
    etl.generate(input_location, output_location)


if __name__ == "__main__":
    generate()
