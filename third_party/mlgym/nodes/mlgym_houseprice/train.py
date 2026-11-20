"""Editable baseline of node mlgym_houseprice (MLGym regressionKaggleHousePrice, Track G port).

The model is that of MLGym's ``data/regressionKaggleHousePrice/baseline.py``
(CC BY-NC 4.0, Meta Platforms): numeric columns mean-imputed and
standardised, object columns imputed with "missing" and one-hot encoded
(unknown categories ignored), Ridge regression with alpha 0.1 on the raw
sale price.  The frozen wrapper loads and splits the data, fits the
estimator returned by ``build_model`` once on the training rows, and scores
it.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALPHA = 0.1


def build_model(x_train, seed):
    """An unfitted estimator for the training features (a DataFrame without Id and SalePrice)."""
    numeric_features = x_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = x_train.select_dtypes(include=["object"]).columns
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return Pipeline(steps=[("preprocess", preprocessor), ("model", Ridge(alpha=ALPHA))])


def transform_target(y):
    return y


def inverse_transform_target(z):
    return z
