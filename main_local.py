#!/usr/bin/env python 

####################
# Required Modules #
####################
import pandas as pd
import sys
import joblib
from sklearn.model_selection import train_test_split

##################
# Configurations #
##################
sys.path.append( './src' )
from src.config import RAW_DATA_PATH, TRAIN_CSV_LOCAL, TEST_CSV_LOCAL, MY_PARAMS, TRAINED_MODEL_PATH_LOCAL
from src.data_pipeline import Preprocessor
from src.model import Model


#################
# Core Function #
#################

# run mlp training pipeline
def run(data_path):
    """Runs the machine learning training pipeline locally.

    Args:
        data_path (str): input data path
   
    """

    # import data
    df = pd.read_csv(data_path)
    print(df.head())
    print(df.shape)
    
    # transform data
    preprocessor = Preprocessor(df)
    clean_engineered_data = preprocessor.transform()
    print(clean_engineered_data)
    print(clean_engineered_data.columns)

    # split data
    train_df, test_df = train_test_split(
        clean_engineered_data, train_size=.8, test_size=.2, shuffle=False
    )

    # export train-test data
    train_df.to_csv(TRAIN_CSV_LOCAL, index=False)
    test_df.to_csv(TEST_CSV_LOCAL, index=False)

    # train model
    my_model = Model()
    print("Current model:", my_model)
    print(my_model.train(df=train_df,params=MY_PARAMS))
    
    # save model
    joblib.dump(my_model, TRAINED_MODEL_PATH_LOCAL)


if __name__ == "__main__":
    run(RAW_DATA_PATH)
