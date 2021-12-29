# build from base image
FROM python:3.7 as base

# install required packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy source codes
WORKDIR mlops
COPY . /mlops

# run the training layer
from base as training_layer
ENTRYPOINT ["python", "/mlops/src/train.py"]

# run the inference layer
from base as inference_layer
ENTRYPOINT ["python", "/mlops/src/inference.py"]

