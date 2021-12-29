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
from sagemaker.predictor import csv_serializer
# from sagemaker_training import environment
# env = environment.Environment()
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

# Define IAM role
role = get_execution_role()
print("role: ", role)
region = boto3.session.Session().region_name # set the region of the instance
print("my_region: ", region)
instance_type = 'local'


# Create the parser
parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
parser.add_argument(
    '--script_type', 
    type=str, 
    default='train',
    help='choose train or inference'
)


args = parser.parse_args()


if __name__ == "__main__":

    if args.script_type == 'train':
#         os.system('docker build  --target training_layer -t aws:train .')
        os.system('docker build --target training_layer -t aws:train .')
        image_uri="aws:train"
            
#         train_input = TrainingInput(
#             "s3://sagemaker-us-east-2-397671599229/data/preprocessed/titanic_train.csv", content_type="csv"
#         )
        
#         validation_input = TrainingInput(
#             "s3://sagemaker-us-east-2-397671599229/data/preprocessed/titanic_test.csv", content_type="csv"
#         )
        
        estimator = Estimator(image_uri=image_uri, 
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
#         model_dir = env.model_dir
#         print(model_dir)
        
        # Deploy model
        aws_sklearn_predictor = estimator.deploy(instance_type='local', 
                                                   initial_instance_count=1)

        # Print the endpoint to test in next step
        print("endpoint: ", aws_sklearn_predictor.endpoint)

    elif args.script_type == 'inference':
        print("inference")

        
#         image_uri="aws:infer"
#         os.system('docker build  --target inference_layer -t aws:infer .')

#         estimator = Estimator(image_uri=image_uri, 
#                             role=role, 
#                             instance_count=1, 
#                             instance_type='local', 
#                             model_uri='file://./model_checkpoint/model.tar.gz')

#         print("Local: Start deploying ... ")
# #         predictor = model.deploy(
# #             initial_instance_count=1,
# #             instance_type='local'
# #         )
#         estimator.fit(inputs={"test": 'file://./data/processed/test/titanic_test.csv'})
#         training_job_name = estimator.latest_training_job.name
#         estimator.attach(training_job_name)
#         predictor = estimator.deploy(initial_instance_count=1,
#                         instance_type='local',
#                          serializer=csv_serializer)
#         print("predictor: ", predictor)

    else:
        print("please try train or inference")