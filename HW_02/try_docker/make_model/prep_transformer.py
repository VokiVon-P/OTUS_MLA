from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class PrepDataTransformer(TransformerMixin, BaseEstimator):
    """Предварительная обработка и заполениени пропусков в данных """

    def __init__(self):
        self.f_trestbps = None
        self.f_chol = None
        self.f_fbs = None
        self.f_restecg = None
        self.f_thalch = None
        self.f_exang = None
        self.f_oldpeak = None
        self.f_slope = None
        self.f_ca = None
        self.f_thal = None
        self.work_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 
        'exang', 'oldpeak', 'slope', 'ca', 'thal']


    def fit(self, X, y=None):
        if type(X) != pd.core.frame.DataFrame:
            return self

        self.f_trestbps = X["trestbps"].mean()
        self.f_chol = X["chol"].mean()
        self.f_fbs = X["fbs"].mode()[0]
        self.f_restecg = X["restecg"].mode()[0]
        self.f_thalch = X["thalch"].mean()
        self.f_exang = X["exang"].mode()[0]
        self.f_oldpeak = X["oldpeak"].mean()
        self.f_slope = X["slope"].mode()[0]
        self.f_ca = X["ca"].mean()
        self.f_thal = X["thal"].mode()[0]

        return self

    def transform(self, X):
        
        if type(X) == pd.core.frame.DataFrame:
            x = X[self.work_columns].copy()
            x["trestbps"].fillna(self.f_trestbps, inplace=True)
            x["chol"].fillna(self.f_chol, inplace=True)
            x["fbs"].fillna(self.f_fbs, inplace=True)
            x["restecg"].fillna(self.f_restecg, inplace=True)
            x["thalch"].fillna(self.f_thalch, inplace=True)
            x["exang"].fillna(self.f_exang, inplace=True)
            x["oldpeak"].fillna(self.f_oldpeak, inplace=True)
            x["slope"].fillna(self.f_slope, inplace=True)
            x["ca"].fillna(self.f_ca, inplace=True)
            x["thal"].fillna(self.f_thal, inplace=True)

        elif type(X) == pd.core.frame.Series:
            x = X[self.work_columns].copy()
            if x["trestbps"] is None:
                x["trestbps"] = self.f_trestbps
            if x["chol"] is None:
                x["chol"] = self.f_chol
            if x["fbs"] is None:
                x["fbs"] = self.f_fbs
            if x["restecg"] is None:
                x["restecg"] = self.f_restecg
            if x["thalch"] is None:
                x["thalch"] = self.f_thalch
            if x["exang"] is None:
                x["exang"] = self.f_exang
            if x["oldpeak"] is None:
                x["oldpeak"] = self.f_oldpeak
            if x["slope"] is None:
                x["slope"] = self.f_slope
            if x["ca"] is None:
                x["ca"] = self.f_ca
            if x["thal"] is None:
                x["thal"] = self.f_thal
            x = x.to_frame().T

        else:
            x = X

        return x
