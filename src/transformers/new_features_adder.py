import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class NewFeaturesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, attribs):
        self.population_index = attribs.index('population')
        self.households_index = attribs.index('households')
        self.total_rooms_index = attribs.index('total_rooms')
        self.total_bedrooms_index = attribs.index('total_bedrooms')
        self.attribs = attribs

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray, y: np.ndarray=None) -> np.ndarray:
        people_per_household = X[:, self.population_index] / X[:, self.households_index]
        bedrooms_per_room = X[:, self.total_bedrooms_index] / X[:, self.total_rooms_index]
        # self.attribs.append('people_per_household')
        # self.attribs.append('bedrooms_per_room')
        self.new_attribs = ['people_per_household', 'bedrooms_per_room']
        return np.c_[X, people_per_household, bedrooms_per_room]
