#!/usr/bin/env python 

####################
# Required Modules #
####################

# Libs
import joblib
import os
import pandas as pd

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
from config import TRAIN_CSV, MODEL_PARAMS, TRAINED_MODEL_PATH 


if __name__ == "__main__":
    # Retrieving feature-engineered data
    df = pd.read_csv(TRAIN_CSV)

    # instantiate the model class
    my_model = Model()
    print("Current model:", my_model)

    # train the model
    roc_score, model_chkpt = my_model.train(df=df,params=MODEL_PARAMS)
    print("roc_score: ", roc_score)
    print("model_chkpt: ", model_chkpt)

    # export the model
    model_dir = os.environ['SM_MODEL_DIR']
    joblib.dump(model_chkpt, os.path.join(model_dir, "model.joblib"))
    print("model training complete!")