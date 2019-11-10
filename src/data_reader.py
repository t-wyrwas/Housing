import os
import pandas as pd
from transformers.index_adder import IndexAdder
from sklearn.model_selection import train_test_split

rootdir = os.path.join('..\\')
datadir = os.path.join(rootdir, 'data')
dataraw = os.path.join(datadir, 'raw')
raw_datafile = os.path.join(dataraw, 'housing.csv')

class DataReader:

    def __init__(self, rootdir: str='../'):
        datadir = os.path.join(rootdir, 'data')
        self.dataraw = os.path.join(datadir, 'raw')
        self.df = None
        self.train_df = None
        self.test_df = None

    def read_data(self, filename: str) -> pd.DataFrame:
        raw_datafile = os.path.join(self.dataraw, filename)
        df = pd.read_csv(raw_datafile)
        index_columns = ['longitude', 'latitude']
        index_adder = IndexAdder(index_columns)
        df = index_adder.fit_transform(df)
        self.df = df
        train_set, test_set = train_test_split(df.values, test_size=0.2, random_state=44)
        self.train_df = pd.DataFrame(train_set, columns=df.columns)
        self.test_df = pd.DataFrame(test_set, columns=df.columns)
        return df
 