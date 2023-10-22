import os 
import sys
import numpy as np
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path , 'wb') as f :
            pickle.dump(obj,f)


    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models:dict):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)

            y_train_preds = model.predict(X_train)

            y_test_preds = model.predict(X_test)

            train_model_score ,text_model_score = r2_score (y_train,y_train_preds) ,r2_score(y_test ,y_test_preds)
            
            report[list(models.keys())[i]] = text_model_score
        
        return report


    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(filepath : str):
    try :
        with open(filepath ,'rb') as f:
            return pickle.load(f)
    
    except Exception as e:
        raise CustomException(e,sys)
