from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation ,DataTransformationConfig 
from src.components.model_trainer import ModelTrainer ,ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_DataIngestion(self):
        logging.info('Entered Data Ingestion Component')
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the Dataset as DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train Test Splitter Initiated')
            train_set,test_set = train_test_split(df ,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion is Completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    o = DataIngestion()
    train_data ,test_data = o.initiate_DataIngestion()
    data_Transformation = DataTransformation()
    train_arr , test_arr ,_ =data_Transformation.initiate_data_transformation(train_data,test_data)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(training_arr=train_arr ,test_arr=test_arr,preprocessor_path=_)
