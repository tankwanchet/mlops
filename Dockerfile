# build from base image
FROM python:3.7

# install required packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy source codes
WORKDIR mlops
COPY . /mlops

# run the training pipeline
ENTRYPOINT ["python", "./src/train.py"]