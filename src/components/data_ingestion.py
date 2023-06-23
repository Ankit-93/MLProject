
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import Custom_Exception
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import Model_Trainer
from dataclasses import dataclass

@dataclass
class DataIngestionConfig():
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','data.csv')

class DataIngestion():

    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Initiated...')
        try:
            df = pd.read_csv('src/notebook/data/StudentsPerformance.csv')
            df.rename(columns={ 'race/ethnicity': 'race_ethnicity',
                                'parental level of education': 'parental_level_of_education',
                                'test preparation course': 'test_preparation_course',
                                'math score': 'math_score',
                                'reading score':'reading_score',
                                'writing score':'writing_score'}, inplace=True)
            
            logging.info('Read the DataSet as Dataframe')
            os.makedirs(os.path.dirname(self.config.train_data_path),exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, header = True)
            logging.info('Raw data stored')
            logging.info('Data Splitting Initiated')
            train_set,test_set = train_test_split(df , test_size=0.2,random_state=42)
            train_set.to_csv(self.config.train_data_path,index =False, header=True)
            test_set.to_csv(self.config.test_data_path,index =False, header=True)
            logging.info('Data Ingestion Completed..')

            return(
                self.config.train_data_path,
                self.config.test_data_path
            )
        except Exception as e:
            raise Custom_Exception(e,sys)



if __name__ == '__main__':
    obj = DataIngestion()
    train,test = obj.initiate_data_ingestion()

    data_transform = DataTransformation()
    train_arr,test_arr,_ = data_transform.initiate_data_transformation(train,test)

    model_trainer = Model_Trainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))

