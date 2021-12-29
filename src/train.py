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
from config import TRAIN_CSV, MY_PARAMS, TRAINED_MODEL_FILENAME 


if __name__ == "__main__":
    # Retrieving feature-engineered data
    df = pd.read_csv(TRAIN_CSV)

    # Testing string representation of Model
    my_model = Model()
    print("Current model:", my_model)
    roc_score, model_chkpt = my_model.train(df=df,params=MY_PARAMS)
    print("roc_score: ", roc_score)
    print("model_chkpt: ", model_chkpt)
    pickle.dump(model_chkpt, open(TRAINED_MODEL_FILENAME, 'wb'))
    print("model training complete!")