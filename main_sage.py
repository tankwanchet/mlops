#!/usr/bin/env python 

####################
# Required Modules #
####################

# Libs
import argparse
import io
import joblib
import os
import sagemaker
from sagemaker.estimator import Estimator
import boto3
import pandas as pd
import pickle
from sagemaker import Model, LocalSession, get_execution_role
from sagemaker.session import TrainingInput

##################
# Configurations #
##################
from src.config import TRAINED_MODEL_PATH, ACCOUNT_ID, ECR_REPOSITORY, TAG, REGION, URI_SUFFIX, S3_PREFIX, FULL_S3_TRAIN_PATH, FULL_S3_TEST_PATH, TRAINED_MODEL, MODEL_PARAMS

# Docker related info
# ACCOUNT_ID = boto3.client("sts").get_caller_identity().get("Account")

# Docker configs
IMAGE_URI = "{}.dkr.ecr.{}.{}/{}:{}".format(ACCOUNT_ID, REGION, URI_SUFFIX, ECR_REPOSITORY, TAG)
PASSWORD_STDIN = "{}.dkr.ecr.{}.{}".format(ACCOUNT_ID, REGION, URI_SUFFIX)

# Data configs
ROLE = get_execution_role()
S3_BUCKET = sagemaker.Session().default_bucket()
S3_KEY_TRAIN = '{}/titanic_train.csv'.format(S3_PREFIX)
S3_KEY_TEST = '{}/titanic_test.csv'.format(S3_PREFIX)

# AWS Credentials & IAM Role
SESSION = sagemaker.Session()
S3_PATH = 's3://{}/{}/'.format(S3_BUCKET, S3_PREFIX)
# region = boto3.session.Session().region_name # set the region of the instance
LOCAL_DATA_PATH = "/home/ec2-user/SageMaker/mlops/data/processed"
LOCAL_TRAIN_PATH = os.path.join(LOCAL_DATA_PATH, "titanic_train.csv")
LOCAL_TEST_PATH = os.path.join(LOCAL_DATA_PATH, "titanic_test.csv")
OUTPUT_PATH = S3_PATH + 'model_output' 

# Checkpointing
bucket=sagemaker.Session().default_bucket()
base_job_name="sagemaker-checkpoint-test"
checkpoint_in_bucket="checkpoints"
CHECKPOINT_S3_BUCKET="s3://{}/{}/{}".format(bucket, base_job_name, checkpoint_in_bucket)
CHECKPOINT_LOCAL_PATH="/opt/ml/checkpoints"

# The S3 URI to store the checkpoints
checkpoint_s3_bucket="s3://{}/{}/{}".format(bucket, base_job_name, checkpoint_in_bucket)


# Argparser
parser = argparse.ArgumentParser(description='List the content of a folder')
parser.add_argument(
    '--script_type', 
    type=str, 
    default='train',
    help='choose train or inference'
)
args = parser.parse_args()

####################
# Helper Functions #
####################

def get_s3_df(bucket, key):
    """Gets data from S3 bucket
  
        Args:
            bucket (str): bucket name
            key (str): subfolder followed by the specific data file name including extension
        
        Returns:
            df (pd.DataFrame): dataframe           
    """
    s3client = boto3.client('s3')
    obj = s3client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    return df

def check_path(s3_bucket, s3_prefix):
    """Checks if the s3 path is valid
        
        Args:
            s3_bucket (str): s3 bucket name
            s3_prefix (str): s3 folder name
            
        Returns:
            exists (str): binary output
    """
    
    s3client = boto3.client('s3')
    result = s3client.list_objects(Bucket=s3_bucket, Prefix=s3_prefix)
    exists=False
    if 'Contents' in result:
        exists=True
    return exists

def set_data(s3_bucket, s3_prefix, s3_data_path, local_data_path):
    """Sets s3 data
    
        Args:
            s3_bucket (str): s3 bucket name
            s3_prefix (str): s3 folder name 
            s3_data_path (str): s3 data path
            local_data_path (str): local data in AWS SageMaker instance
        
        Returns:
            valid_data_path (str): valid s3 data path
    """
    
    # check S3 Data availability 
    check_output = check_path(s3_bucket=s3_bucket, s3_prefix=s3_prefix)
    if check_output:
        valid_data_path = s3_data_path
        print('s3 data path is valid.')
    else: 
        valid_data_path = SESSION.upload_data(local_data_path, key_prefix=s3_prefix)
        
    return valid_data_path

#################
# Core function #
#################

def train():
        """Trains the ML model on an end-to-end pipeline."""
        
        # set Docker env
        try: 
            # pull docker image
            output = os.system('docker pull {}'.format(IMAGE_URI))
            if output != 0:
                # if docker pull fails, build docker image
                os.system('aws ecr get-login-password --region {} | docker login --username AWS --password-stdin {}'.format(
                    REGION, 
                    PASSWORD_STDIN
                )
                         )
                # docker build
                os.system('docker build --no-cache -t {}:{} .'.format(
                    ECR_REPOSITORY, 
                    TAG
                )
                         )
                # tag docker image with ECR
                os.system('docker tag {}:{} {}.dkr.ecr.{}.{}/{}:{}'.format(
                    ECR_REPOSITORY, 
                    TAG, 
                    ACCOUNT_ID, 
                    REGION, 
                    URI_SUFFIX, 
                    ECR_REPOSITORY, 
                    TAG
                )
                         )
                # docker push image
                os.system('docker push {}.dkr.ecr.{}.{}/{}:{}'.format(
                    ACCOUNT_ID, 
                    REGION, 
                    URI_SUFFIX, 
                    ECR_REPOSITORY, 
                    TAG
                )
                         )
        except:
            print('Other Docker errors exist.')

        # check S3 Data availability 
        train_valid_path = set_data(
            s3_bucket=S3_BUCKET, 
            s3_prefix=S3_KEY_TRAIN, 
            s3_data_path=FULL_S3_TRAIN_PATH, 
            local_data_path=LOCAL_TRAIN_PATH
        )
        print('s3 bucket training dataset is availabe.')

        test_valid_path = set_data(
            s3_bucket=S3_BUCKET, 
            s3_prefix=S3_KEY_TEST, 
            s3_data_path=FULL_S3_TEST_PATH, 
            local_data_path=LOCAL_TEST_PATH
        )
        print('s3 bucket testing dataset is availabe.')
        
        # instantiate the estimator
        estimator = Estimator(
            image_uri=IMAGE_URI,
            role=ROLE,
            instance_count=1,
            instance_type='ml.m4.xlarge',
            output_path=OUTPUT_PATH,
            sagemaker_session=SESSION,
            checkpoint_s3_uri=CHECKPOINT_S3_BUCKET,
            checkpoint_local_path=CHECKPOINT_LOCAL_PATH
        )
        
        print("instantiate estimator object")
        
        # set hyperparameters
#         estimator.set_hyperparameters(MODEL_PARAMS)
        
#         print("set hyperparameters for estimator")
        
        # load data
        train_input = sagemaker.TrainingInput(
           train_valid_path, 
            content_type="csv"
        )

        validation_input = sagemaker.TrainingInput(
            test_valid_path, 
            content_type="csv"
        )

        print("Local: Start fitting ... ")
        
        # train the estimator
        estimator.fit(
            inputs={"train": train_input, "validation": validation_input}
        )

        print("Fitting complete")


if __name__ == "__main__":
    
    if args.script_type == "train":
        train()    
        
    elif args.script_type == "infer":
        
        # instantiate the trained model
        trained_model = sagemaker.model.Model(
            model_data=TRAINED_MODEL,
            image_uri=IMAGE_URI,
            role=ROLE
        )  
        
        
        # deploy trained model
        print("deploying...")
        trained_model.deploy(
            initial_instance_count=1, 
            instance_type='ml.m4.xlarge'
        )

        print("infer now!")
    
#     # set Docker env
#     try: 
#         # pull docker image
#         output = os.system('docker pull {}'.format(IMAGE_URI))
#         if output != 0:
#             # if docker pull fails, build docker image
#             os.system('aws ecr get-login-password --region {} | docker login --username AWS --password-stdin {}'.format(REGION, PASSWORD_STDIN))
#             os.system('docker build --no-cache -t {}:{} .'.format(ECR_REPOSITORY, TAG))
#             os.system('docker tag {}:{} {}.dkr.ecr.{}.{}/{}:{}'.format(ECR_REPOSITORY, TAG, ACCOUNT_ID, REGION, URI_SUFFIX, ECR_REPOSITORY, TAG))
#             os.system('docker push {}.dkr.ecr.{}.{}/{}:{}'.format(ACCOUNT_ID, REGION, URI_SUFFIX, ECR_REPOSITORY, TAG))
#     except:
#         print('Other Docker errors exist.')

        
#     # check S3 Data availability 
#     train_valid_path = set_data(
#         s3_bucket=S3_BUCKET, 
#         s3_prefix=S3_KEY_TRAIN, 
#         s3_data_path=FULL_S3_TRAIN_PATH, 
#         local_data_path=LOCAL_TRAIN_PATH
#     )
#     print('s3 bucket training dataset is availabe.')
    
#     test_valid_path = set_data(
#         s3_bucket=S3_BUCKET, 
#         s3_prefix=S3_KEY_TEST, 
#         s3_data_path=FULL_S3_TEST_PATH, 
#         local_data_path=LOCAL_TEST_PATH
#     )
#     print('s3 bucket testing dataset is availabe.')
    
#     estimator = Estimator(
#         image_uri=IMAGE_URI,
#         role=ROLE,
#         instance_count=1,
#         instance_type='ml.m4.xlarge',
#         output_path=OUTPUT_PATH,
#         sagemaker_session=SESSION,
#         checkpoint_s3_uri=CHECKPOINT_S3_BUCKET,
#         checkpoint_local_path=CHECKPOINT_LOCAL_PATH
#     )
    
#     print("instantiate estimator object")
        
#     train_input = sagemaker.TrainingInput(
#        train_valid_path, 
#         content_type="csv"
#     )
    
#     validation_input = sagemaker.TrainingInput(
#         test_valid_path, 
#         content_type="csv"
#     )
    
#     print("Local: Start fitting ... ")
    
#     estimator.fit(
#         inputs={"train": train_input, "validation": validation_input}
#     )
    
#     print("Fitting complete")


#     check_output = check_path(s3_bucket=S3_BUCKET, s3_prefix=S3_KEY_TRAIN)
#     if check_output:
#         train_path = FULL_S3_TRAIN_PATH
#         print('s3 bucket training dataset is availabe.')
#     else: 
#         train_path = SESSION.upload_data(LOCAL_TRAIN_PATH, key_prefix = S3_PREFIX)

    
    
    
#     except:
#         print('Please check if s3 bucket is available.')
    
#     try:
#         check_path(s3_bucket=S3_BUCKET, s3_prefix=S3_KEY_TEST)
#         print('s3 bucket testing dataset is availabe.')

#     except:
#         print('Please check if s3 bucket is available.')
    
        
    
#     train_data = get_s3_df(bucket= S3_BUCKET, key=S3_KEY_TRAIN)
#     test_data = get_s3_df(bucket= S3_BUCKET, key=S3_KEY_TEST)
#     print("train_data: ", test_data)    
    
    # push data to S3 bucket
#     test_path = session.upload_data(local_test_path, key_prefix = S3_PREFIX)
#     train_path = session.upload_data(local_train_path, key_prefix = S3_PREFIX)
#     print('train_path: ', train_path)
#     print('test_path: ', test_path)
    
    
    
    
#     # run Dockerfile
#     os.system('aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 397671599229.dkr.ecr.us-east-2.amazonaws.com')
#     os.system('docker build --no-cache -t mlops .')
#     os.system('docker tag mlops:latest 397671599229.dkr.ecr.us-east-2.amazonaws.com/mlops:latest')
# #     os.system('aws ecr get-login-password | docker login --username AWS --password-stdin 397671599229.dkr.ecr.us-east-2.amazonaws.com')
#     os.system('docker push 397671599229.dkr.ecr.us-east-2.amazonaws.com/mlops:latest')

# #     image_uri="aws:train"
# #     print("byoc_image_uri: ", byoc_image_uri)

#     # image uri
#     image_uri = "397671599229.dkr.ecr.us-east-2.amazonaws.com/mlops:latest"
#     print("image_uri: ", image_uri)
    
    
    
    
    
    
    
    
    
    
    
    
#     estimator = Estimator(
#         image_uri=image_uri, 
#         role=role, 
#         instance_count=1,
#         instance_type='ml.m4.xlarge',
#         output_path=bucket_name, 
#         sagemaker_session=session
# #         base_job_name=base_job_name,
# #         checkpoint_s3_uri=checkpoint_s3_bucket,
# #         checkpoint_local_path=checkpoint_local_path
#     )
    
    # create s3 bucket if unavailable
#     try:
#         if  region == 'us-east-1':
#             s3.create_bucket(Bucket=bucket_name)
#         print('S3 bucket created successfully')
#     except Exception as e:
#         print('S3 error: ',e)

#    print("Local: Start fitting ... ")
#     train_input = sagemaker.TrainingInput(
#         "s3://titanic-training-info/train/titanic_train.csv", 
#         content_type="csv"
#     )
    
#     validation_input = sagemaker.TrainingInput(
#         "s3://titanic-training-info/test/titanic_test.csv", 
#         content_type="csv"
#     )
    
#     estimator.fit(
#         inputs={"train": train_input, "validation": validation_input}
#     )
    
#     print("Fitting complete")
    
# https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-train-model.html
# https://github.com/aws/sagemaker-python-sdk/issues/384
# https://stackoverflow.com/questions/70163549/member-must-satisfy-regular-expression-pattern-httpss3

# https://github.com/snehalnair/SageMaker-in-5-steps/blob/master/Spam-Classifier.ipynb
# https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-console.html
# https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html#byoc-training-step5

# docker issues
# https://stackoverflow.com/questions/34689445/cant-push-image-to-amazon-ecr-fails-with-no-basic-auth-credentials

# data checker
# https://stackoverflow.com/questions/57957585/how-to-check-if-a-particular-directory-exists-in-s3-bucket-using-python-and-boto