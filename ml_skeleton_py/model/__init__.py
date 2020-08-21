"""Model package."""

from .train import train

from .predict import predict_from_file, predict

__all__ = ["train", "predict", "predict_from_file"]
