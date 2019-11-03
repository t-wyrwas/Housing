import os
import pandas as pd
from transformers.index_adder import IndexAdder

rootdir = os.path.join('..\\')
datadir = os.path.join(rootdir, 'data')
dataraw = os.path.join(datadir, 'raw')
raw_datafile = os.path.join(dataraw, 'housing.csv')

class DataReader:

    def __init__(self, rootdir: str='../'):
        datadir = os.path.join(rootdir, 'data')
        self.dataraw = os.path.join(datadir, 'raw')
        self.df = None

    def read_data(self, filename: str) -> pd.DataFrame:
        raw_datafile = os.path.join(self.dataraw, filename)
        df = pd.read_csv(raw_datafile)
        index_columns = ['longitude', 'latitude']
        index_adder = IndexAdder(index_columns)
        self.df = index_adder.fit_transform(df)
        return self.df
 