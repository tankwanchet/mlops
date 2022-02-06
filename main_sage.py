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
from sagemaker.sklearn.model import SKLearnModel
import time
from time import gmtime, strftime


  
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
import json

##################
# Configurations #
##################
from src.config import TRAINED_MODEL_PATH, ACCOUNT_ID, ECR_REPOSITORY, TAG, REGION, URI_SUFFIX, S3_PREFIX, FULL_S3_TRAIN_PATH, FULL_S3_TEST_PATH, TRAINED_MODEL, MODEL_PARAMS, TAG_INFER, TRAIN_LAYER, INFER_LAYER

# Docker related info
# ACCOUNT_ID = boto3.client("sts").get_caller_identity().get("Account")

# Docker configs
IMAGE_URI = "{}.dkr.ecr.{}.{}/{}:{}".format(ACCOUNT_ID, REGION, URI_SUFFIX, ECR_REPOSITORY, TAG)
PASSWORD_STDIN = "{}.dkr.ecr.{}.{}".format(ACCOUNT_ID, REGION, URI_SUFFIX)
IMAGE_URI_INFER = "{}.dkr.ecr.{}.{}/{}:{}".format(ACCOUNT_ID, REGION, URI_SUFFIX, ECR_REPOSITORY, TAG_INFER)

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

# Deployment
client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")

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

def set_docker(
    region, 
    password_stdin, 
    ecr_repository, 
    tag, 
    account_id, 
    uri_suffix, 
    docker_layer, 
    image_uri,
    docker_file_name
):
    """Sets a custom docker image and container"""
    # set Docker env
    try: 
        # pull docker image
        output = os.system('docker pull {}'.format(image_uri))
        if output != 0:
            # if docker pull fails, build docker image
            os.system(
                'aws ecr get-login-password --region {} | docker login --username AWS --password-stdin {}'.format(
                    region, 
                    password_stdin
                )
            )
            # docker build
            if args.script_type == "train":
                os.system(
                    'docker build --no-cache --target {} -t {}:{} -f {} .'.format(
                        docker_layer,
                        ecr_repository, 
                        tag,
                        docker_file_name
                    )
                )
            elif args.script_type == "infer":
                os.system(
                    'docker build --no-cache -t {}:{} -f {} .'.format(
                        ecr_repository, 
                        tag,
                        docker_file_name
                    )
                )
            else:
                print('no docker build.')
            # tag docker image with ECR
            os.system(
                'docker tag {}:{} {}.dkr.ecr.{}.{}/{}:{}'.format(
                    ecr_repository, 
                    tag, 
                    account_id, 
                    region, 
                    uri_suffix, 
                    ecr_repository, 
                    tag
                )
            )
            # docker push image
            os.system(
                'docker push {}.dkr.ecr.{}.{}/{}:{}'.format(
                    account_id, 
                    region, 
                    uri_suffix, 
                    ecr_repository, 
                    tag
                )
            )
    except:
        print('Other Docker errors exist.')

#################
# Core function #
#################

def train():
        """Trains the ML model on an end-to-end pipeline."""
        
        set_docker(
            region=REGION, 
            password_stdin=PASSWORD_STDIN, 
            ecr_repository=ECR_REPOSITORY, 
            tag=TAG, 
            account_id=ACCOUNT_ID, 
            uri_suffix=URI_SUFFIX,
            docker_layer=TRAIN_LAYER,
            image_uri=IMAGE_URI,
            docker_file_name="Dockerfile_train"
        )

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
        set_docker(
            region=REGION, 
            password_stdin=PASSWORD_STDIN, 
            ecr_repository=ECR_REPOSITORY, 
            tag=TAG_INFER, 
            account_id=ACCOUNT_ID, 
            uri_suffix=URI_SUFFIX,
            docker_layer=INFER_LAYER,
            image_uri=IMAGE_URI_INFER,
            docker_file_name="Dockerfile_serve"
        )
        
        trained_model = Model(
            image_uri=IMAGE_URI_INFER,
            role=ROLE,
            model_data=TRAINED_MODEL
        )

        predictor = trained_model.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge',
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
    
    
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