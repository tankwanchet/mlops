#!/usr/bin/env python

########
# Libs #
########
import numpy as np
import pandas as pd
import re
from sklearn import preprocessing

##################
# Configurations #
##################
from .config import RAW_DATA_PATH, NOMINAL, NUMERICAL, ORDINAL
TITLE_RE = re.compile(' ([A-Za-z]+)\.')

class Preprocessor:
    def __init__(
        self,
        df: pd.DataFrame) -> None:
        
        self.df = df


    # remove NA features or records
    def impute_nas(self, df, features):
        """Imputes NAs with numeric imputation
        
        Args:
            df (pd.DataFrame): input dataframe
            features (list): input numeric columns
        
        Returns:
            df (pd.DataFrame): cleaned dataframe

        """
        for feature in features:
            self.df[feature] = self.df[feature].fillna(self.df[feature].mean())
        return self.df

    def get_title(self, name):
        r = TITLE_RE.search(str(name))
        # If the title exists, extract and return it.
        if r:
            return r.group(1)
        return""

    def replace_name_with_title(self, df):
        """Replaces name with title
        
        Args:
            df (pd.DataFrame): input dataframe
        
        Returns:
            df (pd.DataFrame): cleaned dataframe
        """
        df['Title'] = df['Name'].apply(self.get_title)
        df = df.drop("Name", axis=1)
        return df

    def encode_ordinal_feature(self, df, features):
        """Encodes ordinal feature into numeric feature
        
        Args:
            df (pd.DataFrame): input dataframe
            features (list): input numeric columns
        
        Returns:
            df (pd.DataFrame): cleaned dataframe
        """
        le  = preprocessing.LabelEncoder()
        for feature in features:
            df[feature] = le.fit_transform(df[feature])
        return df    
    
    def encode_nominal_feature(self, df, features):
        """Encodes ordinal feature into numeric feature
        
        Args:
            df (pd.DataFrame): input dataframe
            features (list): input numeric columns
        
        Returns:
            df (pd.DataFrame): cleaned dataframe
        """
        df = pd.get_dummies(df, columns=features, drop_first = True)
        return df

    def transform(self):
        df = self.impute_nas(self.df, NUMERICAL)
        df = self.replace_name_with_title(df)
        df = self.encode_ordinal_feature(df, ORDINAL)
        df = self.encode_nominal_feature(df, NOMINAL)
        df = df.set_index("PassengerId")
        return df


# if __name__ == "__main__":
#     df = pd.read_csv(RAW_DATA_PATH)
#     print(df.head())
#     print(df.shape)

#     # Test transformation
#     preprocessor = Preprocessor(df)
#     clean_engineered_data = preprocessor.transform()
#     print(clean_engineered_data)
#     print(clean_engineered_data.columns)