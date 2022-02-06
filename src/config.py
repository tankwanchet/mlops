##################
# Configurations #
##################

# Local data
RAW_DATA_PATH = "./data/raw/titanic.csv"
TRAIN_CSV_LOCAL = "./data/processed/titanic_train.csv" 
TEST_CSV_LOCAL = "./data/processed/titanic_test.csv"

# AWS data
TRAIN_CSV = "/opt/ml/input/data/train/titanic_train.csv"
TEST_CSV = "/opt/ml/input/data/test/titanic_test.csv"

# Preprocessing
ORDINAL = ['Pclass', 'Ticket', 'Cabin', 'Parch', 'SibSp']
NOMINAL = ['Embarked', 'Sex', 'Title']
NUMERICAL = ['Age',  'Fare']

# Model training
TARGET = "Survived"
MODEL_PARAMS = {
            'max_depth': 11, 
            'max_features': 8, 
            'max_leaf_nodes': 27, 
            'n_estimators': 24, 
            'n_jobs': -1, 
            'oob_score': True
        }

TRAINED_MODEL_PATH = "/opt/ml/model/model.tar.gz"
TRAINED_MODEL_PATH_LOCAL = "./model_checkpoint/model.tar.gz"

####################
# AWS Service info #
####################

# DOCKER 
ACCOUNT_ID = "397671599229"
ECR_REPOSITORY = "mlops"
TAG = "train"
TAG_INFER = "infer"
REGION = "us-east-2"
URI_SUFFIX = "amazonaws.com"
TRAIN_LAYER = "train_layer"
INFER_LAYER = "infer_layer"

# S3
S3_PREFIX = 'titanic-dataset' #prefix used for data stored within the bucket
FULL_S3_TRAIN_PATH = 's3://sagemaker-us-east-2-397671599229/titanic-dataset/titanic_train.csv'
FULL_S3_TEST_PATH = 's3://sagemaker-us-east-2-397671599229/titanic-dataset/titanic_test.csv'

# Model
TRAINED_MODEL = 's3://sagemaker-us-east-2-397671599229/titanic-dataset/model_output/mlops-2022-01-31-02-06-48-299/output/model.tar.gz'