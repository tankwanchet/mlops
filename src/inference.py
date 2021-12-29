#!/usr/bin/env python 

####################
# Required Modules #
####################

# Libs
import pandas as pd
import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score

# Custom
try:
    from data_pipeline import Preprocessor
except ModuleNotFoundError:
    from data_pipeline import Preprocessor

try:
    from model import Model
except ModuleNotFoundError:
    from .model import Model

##################
# Configurations #
##################
from config import TEST_CSV, MY_PARAMS, TRAINED_MODEL_FILENAME, TARGET

def predict(df, local_model_path):
    """ Predicts using the trained model

    Args:
        df (pd.DataFrame): input data

    Returns:
        predict_y (pd.DataFrame): predicted data 

    """
    features = df.columns[df.columns != TARGET]
    X_test = df[features]
    Y_test = df[TARGET]
    trained_estimator = pickle.load(open(local_model_path, 'rb'))
    predict_y = trained_estimator.predict(X_test)
    return predict_y


if __name__ == "__main__":
    # Retrieving feature-engineered data
    df = pd.read_csv(TEST_CSV)
    predict(
        df=df,
        local_model_path=TRAINED_MODEL_FILENAME
    )