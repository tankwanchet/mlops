##################
# Configurations #
##################
RAW_DATA_PATH = "./data/raw/titanic.csv"
ORDINAL = ['Pclass', 'Ticket', 'Cabin', 'Parch', 'SibSp']
NOMINAL = ['Embarked', 'Sex', 'Title']
NUMERICAL = ['Age',  'Fare']

MY_PARAMS = {
            'max_depth': 11, 
            'max_features': 8, 
            'max_leaf_nodes': 27, 
            'n_estimators': 24, 
            'n_jobs': -1, 
            'oob_score': True
        }
TRAINED_MODEL_FILENAME = './model_checkpoint/trained_model.sav'

TRAIN_CSV = "./data/processed/titanic_train.csv"
TEST_CSV = "./data/processed/titanic_test.csv"
