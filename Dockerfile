# build from base image
FROM python:3.7 as base

# install required packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy source codes
WORKDIR mlops
COPY . /mlops

# run the preprocessing layer
from base as preprocessing_layer
ENTRYPOINT ["python", "./src/preprocessing.py"]

# run the training layer
from base as preprocessing_layer
ENTRYPOINT ["python", "./src/train.py"]

# run the inference layer
from base as inference_layer
ENTRYPOINT ["python", "./src/inference.py"]

