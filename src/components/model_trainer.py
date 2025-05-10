import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error

import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,model_evaluation
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Initiating training model")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            print(X_train.shape,y_train.shape)
            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost" :XGBRegressor(),
                "SVR" : SVR(),
                "KNN" : KNeighborsRegressor()
            }

            model_reports:pd.DataFrame = model_evaluation(models,X_train,y_train)
            best_model = models.get(model_reports.sort_values("Mean Squared Error").index[0])
            # Fit the best model on the training data
            best_model.fit(X_train, y_train)
            
            # Test the best model on the test data
            y_pred = best_model.predict(X_test)
            
            # Log the evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            logging.info(f"Best Model: {best_model}")
            logging.info(f"Mean Squared Error: {mse}")
            logging.info(f"Mean Absolute Error: {mae}")
            logging.info(f"Root Mean Squared Error: {rmse}")
            
            # Save the best model as a pickle file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    

    ingestion  = DataIngestion()
    train_path,test_path = ingestion.initiate_data_ingestion()
    transform = DataTransformation()
    train_arr,test_arr,X_train,y_train = transform.initiate_data_transformation(train_file_path=train_path,test_file_path=test_path)
    
    trainer  = ModelTrainer()
    trainer.initiate_model_trainer(train_arr,test_arr)