"""Model tests.

Test for correct working and output of an ML model. Add tests here to check
for expected error range.

"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import src.settings as s
import unittest
import joblib


class TestModel(unittest.TestCase):
    def test_model_deserializable(self):
        """Test if a model correctly deserializes.

        Note, this infers models should be part of the git repo.
        """
        model_path = os.path.join(s.MODEL_DIR, "model.joblib")
        if os.path.exists(model_path):
            try:
                joblib.load(model_path)

            except Exception:
                self.fail("the deserialization of the model could not be done")
