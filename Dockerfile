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
COPY main_sage.py /opt/ml/code/
RUN mkdir /opt/ml/model/

# run the training layer
ENV SAGEMAKER_PROGRAM /opt/ml/code/src/train.py
