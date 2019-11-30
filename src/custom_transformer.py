import os
import sys
src = os.path.abspath(os.path.join('.\\'))
if src not in sys.path:
    sys.path.append(src)
from typing import List
    
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from transformers.attributes_selector import AttributesSelector
from transformers.new_features_adder import NewFeaturesAdder


class CustomTransformer(TransformerMixin):

    def __init__(self, target: str, all_attribs: List[str], cat_attribs: List[str] = None):
        self.target = target 
        self.all_attribs = all_attribs
        self.cat_attribs = cat_attribs
        self.processed_df = None
        self._initialize()

    def fit(self, df: pd.DataFrame):
        df = df.copy(deep=True)
        self._full_pipeline.fit_transform(df)
        self._attribs_transformed = self._num_attribs + self._num_pipeline_transformers['new_features'].new_attribs + \
            list(
                self._cat_pipeline_transformers['cat_encoder'].get_feature_names())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_matrix = self._full_pipeline.transform(df)
        self.df = pd.DataFrame(processed_matrix.todense(), columns=self._attribs_transformed)
        self.df.set_index(df.index, inplace=True)
        self.columns = self._attribs_transformed
        self.X = self.df.values
        self.Y = np.array(df[self.target].values)
        return self.df

    def _initialize(self):
        num_attribs = [c for c in self.all_attribs if c not in self.cat_attribs + [self.target]]
        self._num_attribs = num_attribs

        self._num_pipeline_transformers = {'num_selector': AttributesSelector(num_attribs),
                                           'imputer': SimpleImputer(strategy="median"),
                                           'new_features': NewFeaturesAdder(num_attribs),
                                           'scaler': StandardScaler()}
        self._cat_pipeline_transformers = {'cat_selector': AttributesSelector(self.cat_attribs),
                                           'cat_encoder': OneHotEncoder(categories='auto')}

        num_pipeline = Pipeline(list(self._num_pipeline_transformers.items()))
        cat_pipeline = Pipeline(list(self._cat_pipeline_transformers.items()))

        self._full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline),
                                                             ('cat_pipeline', cat_pipeline)])
