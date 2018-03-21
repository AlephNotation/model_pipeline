#Read in packages needed to do the modeling
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor #note: Xcode doesnt support OpenMP multi-threading, which XGBoost uses. To fix this you run 'brew install gcc@5' in the terminal to fix.
import xgboost as xgb

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import preprocessing

from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error, explained_variance_score

seed = 4

preprocessing_pipeline = preprocessing.transformer

