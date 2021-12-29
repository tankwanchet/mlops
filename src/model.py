#!/usr/bin/env python 

####################
# Required Modules #
####################

# Libs
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# Custom
try:
    from data_pipeline import Preprocessor
except ModuleNotFoundError:
    from .data_pipeline import Preprocessor

##################
# Configurations #
##################
from config import RAW_DATA_PATH, MY_PARAMS, TRAINED_MODEL_PATH 

###############
# Model Class #
###############

class Model:
    def __init__(self, model_type="RF", target="Survived"):
        """ Sets and trains model
        
        Args:
            model_type (str): input model type
            target (str): input target variable            
        """
        # Private Attributes
        # self._DEFAULT_PARAMS = {}
        self.model_type = model_type
        self.estimator = None
        self.target = target
        
        # Dynamically instantiate the specified model & store default parameters
        self.set_estimator()
        # self.set_defaults()
        
    def set_estimator(self):
        """ Instantiates an estimator according to the current specified model type as a FUNCTION!
            (NOTE: To use estimator, pass in parameters to it after initialisation)
        """
        try:
            if self.model_type == "LR":
                self.estimator = LogisticRegression()
            elif self.model_type == "DT":
                self.estimator = DecisionTreeClassifier()
            elif self.model_type == "RF":
                self.estimator = RandomForestClassifier()
            else:
                raise ValueError("Unsupported model type specified!")
        except ValueError as v:
            logger.error(f"Model: set_estimator(): Error occurred while instantiating estimator! Error: {v}")
        

    def train(self, df, params=None):
        """ Trains the model

        Args:
            df (pd.DataFrame): input data
            params (dict): input model hyperparameters
        """
        
        # split val and train data
        features = df.columns[df.columns != self.target]
        train_df, val_df = train_test_split(
            df, train_size=.8, test_size=.2, shuffle=False
        )

        # train model
        trained_estimator = self.estimator
        trained_estimator.set_params(**params)
        trained_estimator.fit(train_df[features], train_df[self.target])

        # save model
#         pickle.dump(trained_estimator, open(TRAINED_MODEL_FILENAME, 'wb'))

        # predict on test data
        predict_y = trained_estimator.predict(val_df[features])
        
        # Return a evaluation metric (roc_auc in this case) as a single float so the caller can make use of it
        return roc_auc_score(val_df[self.target], predict_y), trained_estimator

#########
# Tests #
#########

# if __name__ == "__main__":
#     # Retrieving feature-engineered data
#     df = pd.read_csv(RAW_DATA_PATH)
#     preprocessor = Preprocessor(df)
#     data_df = preprocessor.transform()
#     print(data_df)

#     # Testing string representation of Model
#     my_model = Model()
#     print("Current model:", my_model)
#     print(my_model.train(df=data_df,params=MY_PARAMS))