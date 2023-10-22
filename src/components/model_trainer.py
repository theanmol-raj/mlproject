import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor ,RandomForestRegressor ,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object ,evaluate_model



@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,training_arr,test_arr,preprocessor_path):
        try:
            logging.info('Splitting traing and test input Data')
            x_train,y_train,x_test,y_test = (
                training_arr[:,:-1],
                training_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'Random Forest' : RandomForestRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'Linear Regression' : LinearRegression(),
                'K-Neighbors Classifier' : KNeighborsRegressor(),
                'XGBClassifier' : XGBRegressor(),
                'CatBoosting Classifier' : CatBoostRegressor(allow_writing_files=False ,verbose = True),
                'AdaBoost Classifier' : AdaBoostRegressor()
             }
            
            logging.info('Finding the best model')
            model_report : dict = evaluate_model(X_train=x_train,y_train=y_train,X_test = x_test ,y_test =y_test,models=models)

            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best Model Found")
            logging.info(f'Best found model on both training and Test datasets : {best_model_name}')

            # preprocessing_obj = 
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            return  r2_score(y_test,predicted)


        except Exception as e:
            raise CustomException(e,sys)

