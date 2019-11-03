import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AttributesSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.attributes].values
