import os
import sys
src = os.path.abspath(os.path.join('../'))
if src not in sys.path:
    sys.path.append(src)

import numpy as np
from custom_transformer import CustomTransformer
from data_reader import DataReader


if __name__ == '__main__':
    reader = DataReader(rootdir='../')
    reader.read_data('housing.csv')

    cat_attribs = ['ocean_proximity']
    target = 'median_house_value'
    all_attribs = reader.df.columns

    processor = CustomTransformer(target, all_attribs, cat_attribs)
    processor.fit_transform(reader.train_df)