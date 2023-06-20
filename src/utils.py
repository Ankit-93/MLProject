import os
import sys 
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import Custom_Exception
from src.logger import logging


def save_object(file_path,obj):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise Custom_Exception(e,sys)

def evaluate_model(train_X,train_y,test_X,test_y,models):
    try:
        report={}
        for i in range(len(list(models.keys()))):
            model = list(models.values())[i]
            model.fit(train_X,train_y)
            train_pred = model.predict(train_X)
            test_pred = model.predict(test_X)
            train_score = r2_score(train_y,train_pred)
            test_score = r2_score(test_y,test_pred)
            
            report[list(models.keys())[i]] = test_score

            
        return report
    except Exception as e:
        raise Custom_Exception(e,sys)

    