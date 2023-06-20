import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import Custom_Exception
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig():
    model_trainer_file_path = os.path.join('artifacts',"model.pkl")

class Model_Trainer():

    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting Train & Test Input data...')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "RandomForest":RandomForestRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "K Neighbors Method":KNeighborsRegressor(),
                "XGB Regression":XGBRegressor(),
                "CatBoost":CatBoostRegressor(),
                "Adaboost":AdaBoostRegressor()
            }
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(models.values())[list(model_report.values()).index(best_model_score)]
            best_model = best_model_name.fit(X_train,y_train)

            if best_model_score<0.6:
                raise Custom_Exception('No Model Found')

            logging.info(f"Best Model found on train and test data")
            save_object(
                file_path= self.config.model_trainer_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            score = r2_score(y_test,predicted)

            return score

        except Exception as e:
            raise Custom_Exception(e,sys)


