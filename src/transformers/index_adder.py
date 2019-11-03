import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class IndexAdder(BaseEstimator, TransformerMixin):

    def __init__(self, index_attribs):
        if len(index_attribs) != 2:
            raise AttributeError('IndexAdder expects two attributes to create index.')
        self.index_attribs = index_attribs

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X['id'] = (X[self.index_attribs[1]]-1000*X[self.index_attribs[0]])
        X.set_index('id', inplace=True)
        return X
