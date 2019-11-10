import os
import sys
src = os.path.abspath(os.path.join('.\\'))
if src not in sys.path:
    sys.path.append(src)
    
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from transformers.attributes_selector import AttributesSelector
from transformers.new_features_adder import NewFeaturesAdder


class DataProcessor:

    def __init__(self):
        self.processed_df = None

    def process_data(self, df) -> pd.DataFrame:
        cat_attribs = ['ocean_proximity']
        num_attribs = [c for c in df.columns if c not in cat_attribs]

        oh_encoder = OneHotEncoder(categories='auto')
        imputer = SimpleImputer(strategy="median")

        num_pipeline = Pipeline([('num_selector', AttributesSelector(num_attribs)),
                                 ('imputer', imputer),
                                 ('new_features', NewFeaturesAdder(num_attribs)),
                                 ('scaler', StandardScaler())])

        cat_pipeline = Pipeline([('cat_selector', AttributesSelector(cat_attribs)),
                                ('cat_encoder', oh_encoder)])

        full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline),
                                                    ('cat_pipeline', cat_pipeline)])

        processed_matrix = full_pipeline.fit_transform(df)
        new_cols = num_attribs + list(oh_encoder.get_feature_names())
        self.processed_df = pd.DataFrame(processed_matrix.todense(), columns=new_cols)
        self.processed_df.set_index(df.index, inplace=True)
        return self.processed_df
