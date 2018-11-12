"""Model explainability helpers."""
from operator import itemgetter

from pandas import DataFrame
import numpy as np
import shap


class TreeExplainer(object):
    """Tree model explainability functions."""

    def __init__(self, model, feature_names=None):
        """
        Create explainer for given model.

        :param model:
        :param feature_names:
        """
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names

    def get_feature_importance(self, observations):
        """
        Calculate shap for features, using given observation(s).

        :param observations: array or DataFrame: One or more observations.
        :return: Pairs (feature, shap_value)
        """
        if isinstance(observations, DataFrame):
            feature_names = observations.columns.values.tolist()
            if self.feature_names is not None and feature_names != self.feature_names:
                raise Exception('DataFrame columns do not match feature names.')

        if isinstance(observations, list):
            if self.feature_names is None:
                raise Exception('Feature names not defined for explainer.'
                                'Pass observation as a dataframe (feature names as column names).')
            observations = np.array(observations)
            if observations.ndim == 1:
                observations = observations.reshape(1, -1)
            feature_names = self.feature_names

        shap_values = self.explainer.shap_values(observations)
        avg_abs_shaps = []
        for i in range(shap_values.shape[1]):
            avg_abs_shaps.append(np.average(np.abs(shap_values[:, i])))
        result = zip(feature_names, avg_abs_shaps)
        return sorted(result, key=itemgetter(1), reverse=True)
