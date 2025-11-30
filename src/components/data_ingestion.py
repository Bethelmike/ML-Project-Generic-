import os
import sys
from typing import Tuple   
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> pd.DataFrame:
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv(r'C:\Users\HIGH HORIZON\Desktop\ML Project 2\src\notebooks\stud.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

)

        except Exception as e:
            logging.error("Error occurred in data ingestion")
            raise CustomException(e, sys)
if __name__ == "__main__":
    obj = DataIngestion()
    tr, ts = obj.initiate_data_ingestion()
    print("Created:", tr, ts)
    # later after data ingestion we will do data transformation
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path=tr, test_path=ts)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

        
