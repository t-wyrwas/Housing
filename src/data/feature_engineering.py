from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def enc_cat_features_one_hot(df: pd.DataFrame, feature_names: List[str]):

    encoder = OneHotEncoder(categories='auto')

    for name in feature_names:
        print(name)
        feature = df[name]
        enc_feature = encoder.fit_transform(feature.to_numpy().reshape(-1,1))
        categories = encoder.categories_[0].tolist()
        enc_df = pd.DataFrame(enc_feature.toarray(), columns=categories, index=df.index)
        df[categories] = enc_df
        df.drop(name, axis=1, inplace=True)
