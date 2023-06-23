import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import Custom_Exception
from src.utils import save_object
from dataclasses import dataclass


@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_datatransformation_object(self):
        ' This function is responsible for data transformation'
        try:
            numerical_columns = ['reading_score','writing_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline = Pipeline(
                steps=[("imputer",SimpleImputer(strategy='median')),
                       ('scaler',StandardScaler(with_mean=False))])
            logging.info('Numerical Columns scaling completed...')
            catergorical_pipeline = Pipeline(
                steps=
                [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categorical Columns encoding completed...')
            preprocessor = ColumnTransformer(
                [
                    
                    ('categorical_pipeline',catergorical_pipeline,categorical_columns),
                    ('numerical_pipeline',num_pipeline,numerical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise Custom_Exception(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading Train & Test Data Completed')
            logging.info('Obtaining Pre Processing Object')
            preprocessor_obj = self.get_datatransformation_object()
            target_column = 'math_score'
            numerical_col = ['reading_score','writing_score']
            input_feature_train = train_df.drop([target_column],axis=1)
            target_train = train_df[target_column]
            input_feature_test = test_df.drop([target_column],axis=1)
            target_test = test_df[target_column]

            logging.info(f'Applying preprocessing on train & test dataframe')
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test)

            train_arr = np.c_[input_feature_train_arr,np.array(target_train)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_test)]
            logging.info('Saving preprocessing abject...')
            save_object(
                file_path = self.config.preprocessor_obj_file_path,
                obj = preprocessor_obj

            )

            return(
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise Custom_Exception(e,sys)



