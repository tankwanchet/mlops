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
    from .data_pipeline import Preprocessor

try:
    from model import Model
except ModuleNotFoundError:
    from .model import Model

##################
# Configurations #
##################
from config import RAW_DATA_PATH, MY_PARAMS, TRAINED_MODEL_FILENAME 


if __name__ == "__main__":
    # Retrieving feature-engineered data
    df = pd.read_csv(RAW_DATA_PATH)
    preprocessor = Preprocessor(df)
    data_df = preprocessor.transform()
    
    # split the data into training and testing datasets
    train_df, test_df = train_test_split(
        df, 
        train_size=.8, 
        test_size=.2, 
        shuffle=False
    )

    # export df as csvs
    train_df.to_csv("titanic_train.csv")
    test_df.to_csv("titanic_test.csv")