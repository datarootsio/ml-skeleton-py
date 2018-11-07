from sklearn.externals import joblib
import pandas as pd
from mlmonkey.metadata import ModelMetadata
import shap


class ModelExplainer(object):
    """Expose model explainability functions."""

    def __init__(self, metadata):
        """
        Load model, data and create explainer.
        """
        if not isinstance(metadata, ModelMetadata):
            raise Exception('Invalid metadata format.')

        self.model = joblib.load(metadata['model_location'])
        self.train_data = pd.read_csv(metadata['input_data_location'])\
            .drop(columns=['species', 'petal_length'])
        self.explainer = shap.TreeExplainer(self.model)

    def get_ex(self):
        return self.explainer

    def plot_feature_importance(self):
        """Plots features importance, as bar plot."""
        shap_values = self.explainer.shap_values(self.train_data)
        shap.summary_plot(shap_values, self.train_data, plot_type='bar')


    def explain_prediction(self, dataframe):
        """Visualize explanation after calculating prediction(s) for given observation(s)."""
        shaps = self.explainer.shap_values(dataframe)
        return shap.force_plot(self.explainer.expected_value, shaps, dataframe)


