"""Model package."""

from .train import (
    train,
)

from .predict import (
    predict_from_file,
)

__all__ = [
    'train',
    'predict_from_file',
]
