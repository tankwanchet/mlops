# build from base image
FROM python:3.7

# install required packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy source codes
COPY src /opt/ml/code/

# run the training pipeline
ENTRYPOINT ["python", "/opt/ml/code/train.py"]