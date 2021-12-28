import os
import sagemaker
from sagemaker.estimator import Estimator
import boto3
import pandas as pd
from sagemaker import Model, LocalSession, get_execution_role

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}
bucket_name = sagemaker_session.default_bucket()
print("bucket: ", bucket_name)
# DUMMY_IAM_ROLE = 'arn:aws:sagemaker:us-east-2:397671599229:notebook-instance/aws-training'
# print('DUMMY_IAM_ROLE: ', DUMMY_IAM_ROLE)
s3 = boto3.client('s3')

# Define IAM role
role = get_execution_role()
print("role: ", role)
region = boto3.session.Session().region_name # set the region of the instance
print("my_region: ", region)
instance_type = 'local'

# Credentials
# role=os.getenv("AWS_SM_ROLE")
# print("role: ", role)
# aws_id=os.getenv("AWS_ID")
# region=os.getenv("AWS_REGION")
image_uri="aws:train"
# print("Training image uri:{}".format(image_uri))
# instance_type=os.getenv("AWS_DEFAULT_INSTANCE")
# bucket_name = os.getenv("AWS_BUCKET")

os.system('docker build -t aws:train .')
estimator = Estimator(image_uri=image_uri, 
                      role=role, 
                      instance_count=1, 
                      instance_type='local', 
                      output_path='file://./model_checkpoint', 
                      base_job_name='mlops_training')

print("Local: Start fitting ... ")
estimator.fit(inputs=f'file://./data/raw/titanic.csv',job_name='mlops_training_1')