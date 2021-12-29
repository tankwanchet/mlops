##################
# Configurations #
##################
RAW_DATA_PATH = "./data/raw/titanic.csv"
ORDINAL = ['Pclass', 'Ticket', 'Cabin', 'Parch', 'SibSp']
NOMINAL = ['Embarked', 'Sex', 'Title']
NUMERICAL = ['Age',  'Fare']
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

TRAIN_CSV = "/opt/ml/input/data/train/titanic_train.csv"
TEST_CSV = "/opt/ml/input/data/test/titanic_test.csv"
