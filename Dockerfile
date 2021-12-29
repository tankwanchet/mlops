# build from base image
FROM python:3.7 as base

# install required packages
RUN pip3 install sagemaker-training
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy source codes
COPY ./data/processed/train/ /opt/ml/input/data/train/
COPY ./src /opt/ml/code/src/
COPY main.py /opt/ml/code/
RUN mkdir /opt/ml/model/

# Defines train.py as script entry point
# ENV SAGEMAKER_PROGRAM /opt/ml/code/src/train.py

# # run the training layer
from base as training_layer
# ENTRYPOINT ["python", "/mlops/src/train.py"]
ENV SAGEMAKER_PROGRAM /opt/ml/code/src/train.py

# # run the inference layer
from base as inference_layer
COPY ./model_checkpoint/model.tar.gz /opt/ml/model/model.tar.gz
ENV SAGEMAKER_PROGRAM /opt/ml/code/src/inference.py
# ENTRYPOINT ["python", "/mlops/src/inference.py"]

