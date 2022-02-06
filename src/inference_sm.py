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

import json


def model_fn(model_dir):
    """
    Deserialize fitted model
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(request_body, request_content_type):
    """
    input_fn
        request_body: The body of the request sent to the model.
        request_content_type: (string) specifies the format/variable type of the request
    """
    if request_content_type == 'application/json':
        request_body = json.loads(request_body)
        inpVar = request_body['Input']
        return inpVar
    else:
        raise ValueError("This model only supports application/json input")


def predict_fn(input_data, model):
    """
    predict_fn
        input_data: returned array from input_fn above
        model (sklearn model) returned model loaded from model_fn above
    """
    return model.predict(input_data)


def output_fn(prediction, content_type):
    """
    output_fn
        prediction: the returned value from predict_fn above
        content_type: the content type the endpoint expects to be returned. Ex: JSON, string
    """
    res = int(prediction[0])
    respJSON = {'Output': res}
    return respJSON




# def predict_fn(df, model):
#     """Predicts based on test dataframe."""
    
#     features = df.columns[df.columns != TARGET]
#     X_test = df[features]
#     Y_test = df[TARGET]
    
#     print("calling model")
#     predictions = model.predict(X_test)
#     return predictions


# def model_fn(model_dir):
#     """Loads the trained model."""
#     print("loading model.joblib from: {}".format(model_dir))
#     tar = tarfile.open(model_dir)
#     trained_estimator = joblib.load(tar.extractfile(member=tar.getmember(name="model.joblib")))
#     return loaded_model

# def model_fn(model_dir):
#     return joblib.load(os.path.join(model_dir, "model.joblib"))



# if __name__ == "__main__":
#     # Retrieving feature-engineered data
#     df = pd.read_csv(TEST_CSV)
#     loaded_model = model_fn(model_dir=TRAINED_MODEL_PATH)
#     pred_y = predict_fn(df=df,model=loaded_model)
#     print("inference completed")