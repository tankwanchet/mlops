#!/usr/bin/env python 

####################
# Required Modules #
####################

# Libs
import argparse
import joblib
import os
import sagemaker
from sagemaker.estimator import Estimator
import boto3
import pandas as pd
from sagemaker import Model, LocalSession, get_execution_role
from sagemaker.session import TrainingInput

##################
# Configurations #
##################
from src.config import TRAINED_MODEL_PATH

# Credentials
sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}
bucket_name = sagemaker_session.default_bucket()
base_job_name = "checkpoint-test"
print("bucket: ", bucket_name)
s3 = boto3.client('s3')

# IAM role
role = get_execution_role()
print("role: ", role)
region = boto3.session.Session().region_name # set the region of the instance
print("my_region: ", region)
instance_type = 'local'

# Argparser
parser = argparse.ArgumentParser(description='List the content of a folder')
parser.add_argument(
    '--script_type', 
    type=str, 
    default='train',
    help='choose train or inference'
)
args = parser.parse_args()


if __name__ == "__main__":


    os.system('docker build -t aws:train .')
    image_uri="aws:train"
        
#         train_input = TrainingInput(
#             "s3://sagemaker-us-east-2-397671599229/data/preprocessed/titanic_train.csv", content_type="csv"
#         )
    
#         validation_input = TrainingInput(
#             "s3://sagemaker-us-east-2-397671599229/data/preprocessed/titanic_test.csv", content_type="csv"
#         )
    
    estimator = Estimator(
        image_uri=image_uri, 
        role=role, 
        instance_count=1, 
        instance_type='local', 
        output_path='file://./model_checkpoint', 
        base_job_name=base_job_name        
    )

    print("Local: Start fitting ... ")
    estimator.fit(
        inputs={"train": 'file://./data/processed/train/titanic_train.csv'} 
    )
    print("Fitting complete")