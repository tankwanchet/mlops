#!/usr/bin/env python 

####################
# Required Modules #
####################

# Libs
import joblib
import os
import pandas as pd
import tarfile

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
from config import TEST_CSV, MODEL_PARAMS, TRAINED_MODEL_PATH, TARGET

###################
# Helper Function #
###################

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
    
    tar = tarfile.open(local_model_path)
    trained_estimator = joblib.load(tar.extractfile(member=tar.getmember(name="model.joblib")))
    predict_y = trained_estimator.predict(X_test)
    return predict_y


if __name__ == "__main__":
    # Retrieving feature-engineered data
    df = pd.read_csv(TEST_CSV)
    pred_y = predict(df=df,local_model_path=TRAINED_MODEL_PATH)
    print("inference completed")