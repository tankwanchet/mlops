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
MY_PARAMS = {
            'max_depth': 11, 
            'max_features': 8, 
            'max_leaf_nodes': 27, 
            'n_estimators': 24, 
            'n_jobs': -1, 
            'oob_score': True
        }
TRAINED_MODEL_PATH = "/opt/ml/model/model.tar.gz"
TRAINED_MODEL_PATH_LOCAL = "./model_checkpoint/model.tar.gz"