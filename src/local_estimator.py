import os
from sagemaker.estimator import Estimator

# Credentials
role=os.getenv("AWS_SM_ROLE")
print("role: ", role)
aws_id=os.getenv("AWS_ID")
region=os.getenv("AWS_REGION")
image_uri="{}.dkr.ecr.{}.amazonaws.com/aws-train".format(aws_id, region)
print("Training image uri:{}".format(image_uri))
instance_type=os.getenv("AWS_DEFAULT_INSTANCE")
bucket_name = os.getenv("AWS_BUCKET")

os.system('docker build -t mlops_env:training .')
estimator = Estimator(image_uri='mlops_env:training', 
                      role=role, 
                      instance_count=1, 
                      instance_type='local', 
                      output_path='/home/chet/projects/mlops/model_checkpoint', 
                      base_job_name='mlops_training')

print("Local: Start fitting ... ")
estimator.fit(inputs=f'file://home/chet/projects/mlops/data/raw/titanic.csv',job_name='mlops_training_1')