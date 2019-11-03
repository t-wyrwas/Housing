import numpy as np
import pandas as pd
import data.feature_engineering as fe

def test_enc_cat_features_one_hot_adds_proper_columns():
    #Arrange
    data = [
        {'cat_f': 'dog', 'cat_f_2': 'plane', 'not_cat': 5},
        {'cat_f': 'cat', 'cat_f_2': 'car', 'not_cat': 8},
        {'cat_f': 'mouse', 'cat_f_2': 'plane', 'not_cat': -7}
    ]
    df = pd.DataFrame(data)
    df.reset_index()

    # Act
    fe.enc_cat_features_one_hot(df, feature_names=['cat_f', 'cat_f_2'])

    # Assert
    columns = df.columns.tolist()
    assert 'dog' in columns
    assert 'cat' in columns
    assert 'mouse' in columns
    assert 'plane' in columns
    assert 'car' in columns

def test_enc_cat_features_one_hot_removes_original_features():
    #Arrange
    data = [
        {'cat_f': 'dog', 'cat_f_2': 'plane', 'not_cat': 5},
        {'cat_f': 'cat', 'cat_f_2': 'car', 'not_cat': 8},
        {'cat_f': 'mouse', 'cat_f_2': 'plane', 'not_cat': -7}
    ]
    df = pd.DataFrame(data)
    df.reset_index()

    # Act
    fe.enc_cat_features_one_hot(df, feature_names=['cat_f', 'cat_f_2'])

    # Assert
    columns = df.columns.tolist()
    assert 'cat_f' not in columns
    assert 'cat_f_2' not in columns

def test_enc_cat_features_one_hot_sets_correct_values():
    #Arrange
    data = [
        {'cat_f': 'dog', 'cat_f_2': 'plane', 'not_cat': 5},
        {'cat_f': 'cat', 'cat_f_2': 'car', 'not_cat': 8},
        {'cat_f': 'mouse', 'cat_f_2': 'plane', 'not_cat': -7}
    ]
    df = pd.DataFrame(data)
    df.reset_index()

    # Act
    fe.enc_cat_features_one_hot(df, feature_names=['cat_f', 'cat_f_2'])

    # Assert
    assert df.iloc[0]['dog'] == 1
    assert df.iloc[0]['cat'] == 0
    assert df.iloc[0]['mouse'] == 0

    assert df.iloc[1]['dog'] == 0
    assert df.iloc[1]['cat'] == 1
    assert df.iloc[1]['mouse'] == 0

    assert df.iloc[2]['dog'] == 0
    assert df.iloc[2]['cat'] == 0
    assert df.iloc[2]['mouse'] == 1

    assert df.iloc[0]['plane'] == 1
    assert df.iloc[0]['car'] == 0

    assert df.iloc[1]['plane'] == 0
    assert df.iloc[1]['car'] == 1

    assert df.iloc[2]['plane'] == 1
    assert df.iloc[2]['car'] == 0
