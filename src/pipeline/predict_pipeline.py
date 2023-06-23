import sys
import pandas as pd
from src.exception import Custom_Exception
from src.utils import load_object

class PredictPipeline:

    def __init__(self):
        pass

    def prediction(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print(features)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise Custom_Exception(e,sys)


class CustomData:

    def __init__(self,
            gender,
            race_ethnicity,
            parental_level_of_education,
            lunch,
            test_preparation_course,
            reading_score,
            writing_score):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_dataframe(self):

        try:
            custom_dataframe={
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                "lunch":[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }

            return pd.DataFrame(custom_dataframe)
        except Exception as e:
            raise Custom_Exception(e,sys)
        
