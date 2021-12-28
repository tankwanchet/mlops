#!/usr/bin/env python 

####################
# Required Modules #
####################

# Libs
import argparse
import os
import sagemaker
from sagemaker.estimator import Estimator
import boto3
import pandas as pd
from sagemaker import Model, LocalSession, get_execution_role
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

# Credentials
sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}
bucket_name = sagemaker_session.default_bucket()
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
    help='choose process, train or inference'
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
                            base_job_name='mlops_training')

        print("Local: Start fitting ... ")
        estimator.fit(inputs=f'file://./data/raw/titanic.csv',job_name='mlops_training_1')

    elif args.script_type == 'inference':
        os.system('docker build  --target training_layer -t aws:infer .')
        image_uri="aws:infer"
        estimator = Estimator(image_uri=image_uri, 
                            role=role, 
                            instance_count=1, 
                            instance_type='local', 
                            output_path='file://./model_checkpoint', 
                            base_job_name='mlops_training')

        print("Local: Start fitting ... ")
        estimator.fit(inputs=f'file://./data/raw/titanic.csv',job_name='mlops_training_1')

    else:
        print("please try preprocess, train or inference")