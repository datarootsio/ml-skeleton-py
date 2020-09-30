""" Configuration file.
All static variables can be assigned in this settings.py file
"""

import os

# Directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")

DATA_RAW = os.path.join(DATA_DIR, "raw")

DATA_TRANSFORMED = os.path.join(DATA_DIR, "transformed")

DATA_STAGING = os.path.join(DATA_DIR, "staging")

DATA_PREDICTIONS = os.path.join(DATA_DIR, "predictions")

ETL_DIR = os.path.join(ROOT_DIR, "ml_skeleton_py", "etl")

MODEL_DIR = os.path.join(ROOT_DIR, "models")

MODEL_METADATA_DIR = os.path.join(ROOT_DIR, "models", "metadata")

# Model Variables
TARGET_VARIABLE = "Class"

DATASET_NAME = "creditcard.csv"

# ---------- PYTEST VARIABLES ---------
# TEST Variables
TEST_DATASET_NAME = "sample_creditcard.csv"

# TEST DIRECTORIES
ASSETS_DIR = os.path.join(ROOT_DIR, "tests", "assets")

EXPECTED_TEMP_TRANSFORMED_DATA_LOC = os.path.join(
    ASSETS_DIR, "transformed", "temp_sample_creditcard.csv"
)
EXPECTED_TRANSFORMED_DATA_LOC = os.path.join(
    ASSETS_DIR, "transformed", TEST_DATASET_NAME
)
UNEXPECTED_TRANSFORMED_DATA_LOC = os.path.join(ASSETS_DIR, "transformed", "dummy.csv")

EXPECTED_RAW_DATA_LOC = os.path.join(ASSETS_DIR, "raw", TEST_DATASET_NAME)
UNEXPECTED_RAW_DATA_LOC = os.path.join(ASSETS_DIR, "raw", "dummy.csv")

EXPECTED_MODEL_LOC = os.path.join(ASSETS_DIR, "models")
