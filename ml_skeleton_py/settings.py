"""Configuration file.
Most of these variables can be overridden through environment variables.
"""

import os
from os.path import abspath

DATA_RAW = os.path.join(abspath(os.getenv("DATA_RAW", "./data/raw/")), "")
DATA_TRANSFORMED = os.path.join(
    abspath(os.getenv("DATA_TRANSFORMED", "./data/transformed/")), ""
)
DATA_STAGING = os.path.join(
    abspath(os.getenv("DATA_TRANSFORMED", "./data/staging/")), ""
)
DATA_PREDICTIONS = os.path.join(
    abspath(os.getenv("DATA_TRANSFORMED", "./data/predictions/")), ""
)

ETL_DIR = os.path.join(abspath(os.getenv("ETL_DIR", "./etl")), "")


MODEL_DIR = os.path.join(abspath(os.getenv("MODEL_DIR", "./models")), "")
MODEL_METADATA_DIR = os.path.join(
    abspath(os.getenv("MODEL_METADATA_DIR", "./models/metadata")), ""
)

FLASK_ENDPOINT_PORT = os.getenv("FLASK_ENDPOINT_PORT", 5000)
