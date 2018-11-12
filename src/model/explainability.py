"""Model explainability helpers."""
from operator import itemgetter

import matplotlib as mpl
from pandas import DataFrame
import numpy as np
mpl.use('TkAgg')
import shap


class TreeExplainer(object):
    """Tree model explainability functions."""

    def __init__(self, model):
        """
        Create explainer for given model.

        :param model:
        """
        self.explainer = shap.TreeExplainer(model)


    def get_shap_values(self, observations):
        """
        Calculate shap for features, using given observation(s).

        :param observations: pandas dataframe: One or more observations.
        Column names of dataframe are feature names.
        :return: Pairs (feature, shap_value)
        """
        if not isinstance(observations, DataFrame):
            raise Exception('Observation must be provided as a pandas DataFrame.')

        shap_values = self.explainer.shap_values(observations)
        avg_abs_shaps = []
        for i in range(shap_values.shape[1]):
            avg_abs_shaps.append(np.average(np.abs(shap_values[:, i])))
        result = zip(observations.columns.values.tolist(), avg_abs_shaps)
        return sorted(result, key=itemgetter(1), reverse=True)