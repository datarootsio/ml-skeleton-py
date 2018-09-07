"""Model tests.

Test for correct working and output of an ML model. Add tests here to check
for expected error range.

"""
import os

def test_model_deserializable():
    """Test if a model correctly deserializes.

    Note, this infers models should be part of the git repo.
    TODO: add tangible examples
    """
    assert os.path.exists('models/')
