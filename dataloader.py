"""
Module for loading and preprocessing the UNSW-NB15 dataset assuming the CSV files
from https://research.unsw.edu.au/projects/unsw-nb15-dataset are downloaded to the
data/ folder of the working directory
"""

import typing
import pandas as pd
import numpy as np
from sklearn import preprocessing as skp


def encode_categoricals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the categorical columns in data as integers using enumeration

    Arguments:
    - data: pandas DataFrame to encode
    """
    categorical_cols = [c for c in data if data[c].dtype == "object"]
    data[categorical_cols] = skp.OrdinalEncoder().fit_transform(data[categorical_cols])


def byterize(x: float | typing.Collection[float]) -> float | typing.Collection[float]:
    """
    Return the number of bits in the nearest byte that can contain x

    Arguments:
    - x: Number or collection of numbers to see the number containing bits round to the byte
    """
    return 2**np.ceil(np.log2(x))


def min_max_scale(
    x: float | typing.Collection[float],
    min_val: float,
    max_val: float
) -> float | typing.Collection[float]:
    """
    Scale x into the range [0, 1] uniformly where x = 1 means before scaling x = max_val,
    and x = 0 means before scaling x = min_val

    Arguments:
    - x: floating point or colleciton of floating points to scale into the range [0, 1]
    - min_val: the minimum value to use for scaling
    - max_val: the maximum value to use for scaling
    """
    return (x - min_val) / (max_val - min_val)


def normalize(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Normalize all the data into the [0, 1] according to the training data's properties.
    Operates inplace

    Arguments:
    - train_data: Training data in a pandas DataFrame containing only numerical values
    - test_data: Testing data in a pandas DataFrame containing only numerical values
    """
    for col in train_data:
        min_val = np.floor(train_data[col].min())
        max_val = byterize(train_data[col].max())
        test_data[col] = test_data[col].clip(upper=max_val)
        train_data[col] = min_max_scale(train_data[col], min_val, max_val)
        test_data[col] = min_max_scale(test_data[col], min_val, max_val)


def extract_X_Y(data: pd.DataFrame) -> typing.Tuple[np.typing.NDArray, np.typing.NDArray]:
    """
    Extract the samples and labels from a dataframe into the numpy arrays X, Y respectively

    Arguments:
    - data: pandas dataframe containing the data samples and the labels in a column named "label"
    """
    Y = data.label.to_numpy()
    X = data.drop(columns="label").to_numpy()
    return X, Y


def load_data() -> typing.Tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
    """
    Load and process the UNSW-NB15 dataset from the CSV files to 4 numpy arrays:
    the training samples, training labels, testing samples, and testing labels
    """
    train_data = pd.read_csv("data/UNSW_NB15_training-set.csv")
    test_data = pd.read_csv("data/UNSW_NB15_testing-set.csv")
    train_data = train_data.drop(columns=["id", "attack_cat"])
    test_data = test_data.drop(columns=["id", "attack_cat"])

    encode_categoricals(train_data)
    encode_categoricals(test_data)

    normalize(train_data, test_data)

    X_train, Y_train = extract_X_Y(train_data)
    X_test, Y_test = extract_X_Y(test_data)
    return X_train, Y_train, X_test, Y_test
