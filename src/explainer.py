import numpy as np

class FeatureAttributionExplainer:
    """Lightweight explainer using model feature importances or coefficients."""
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def shap_values(self, X_trans):
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = self.model.coef_[0]
        else:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)
        return X_trans * importances
