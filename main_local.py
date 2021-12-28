#!/usr/bin/env python 

####################
# Required Modules #
####################
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

##################
# Configurations #
##################
from src.config import RAW_DATA_PATH, TRAIN_CSV, TEST_CSV, MY_PARAMS, TRAINED_MODEL_FILENAME
from src.data_pipeline import Preprocessor
from src.model import Model

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
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    # train model
    my_model = Model()
    print("Current model:", my_model)
    print(my_model.train(df=train_df,params=MY_PARAMS))
    # save model
    pickle.dump(my_model, open(TRAINED_MODEL_FILENAME, 'wb'))

if __name__ == "__main__":
    run(RAW_DATA_PATH)
