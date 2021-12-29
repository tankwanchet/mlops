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

##################
# Configurations #
##################
from src.config import TRAINED_MODEL_FILENAME

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
        os.system('docker build  --target training_layer -t aws:train .')
        image_uri="aws:train"
        estimator = Estimator(image_uri=image_uri, 
                            role=role, 
                            instance_count=1, 
                            instance_type='local', 
                            output_path='file://./model_checkpoint', 
                            base_job_name=base_job_name)

        print("Local: Start fitting ... ")
        estimator.fit(job_name='mlops_training_1')
#         path = os.path.join(TRAINED_MODEL_FILENAME)
#         joblib.dump(estimator, path)
#         print('model persisted at ' + path)


    elif args.script_type == 'inference':
#         os.system('docker build  --target inference_layer -t aws:infer .')
        image_uri="aws:infer"
        os.system('docker build  --target inference_layer -t aws:infer .')

        estimator = Estimator(image_uri=image_uri, 
                            role=role, 
                            instance_count=1, 
                            instance_type='local', 
                            model_uri='file://./model_checkpoint/model.tar.gz')

        print("Local: Start deploying ... ")
#         predictor = model.deploy(
#             initial_instance_count=1,
#             instance_type='local'
#         )
        estimator.fit(wait=False)
        training_job_name = estimator.latest_training_job.name
        estimator.attach(training_job_name)
        estimator.deploy(initial_instance_count=1,
                        instance_type='local',
                         serializer=csv_serializer)

    else:
        print("please try train or inference")