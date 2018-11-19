"""Model explainability helpers."""
from operator import itemgetter

from pandas import DataFrame
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
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

    def feature_importance_for_batch(self, observations):
        """
        Calculate average shap for features.

        :param observations: array or DataFrame: One or more observations.
        :return: Pairs (feature, shap_value)
        """
        shaps_df = self.feature_importance_per_observation(observations)
        means = shaps_df.mean()
        result = zip(shaps_df.columns, means)
        return sorted(result, key=itemgetter(1), reverse=True)

    def feature_importance_per_observation(self, observations):
        """
        Calculate shap for features, per each observation.

        :param observations: array or DataFrame: One or more observations.
        :return: DataFrame of importance scores, row per observation, column per feature
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
        shap_values = np.abs(shap_values)
        return DataFrame(shap_values, columns=feature_names)
