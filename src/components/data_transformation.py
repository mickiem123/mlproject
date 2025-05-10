import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from src.utils import save_object
import pickle
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    target_column_name = 'math_score'
    numerical_features = ['writing_score', 'reading_score']
    categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    transformed_train_file_path = os.path.join('artifacts', 'train_transformed.csv')
    transformed_test_file_path = os.path.join('artifacts', 'test_transformed.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self,numerical_features, categorical_features):
        try:
            logging.info("Data transformation initiated")
            num_transformer = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy = 'median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_transformer = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent'))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_transformer,numerical_features),
                    ('cat',cat_transformer,categorical_features)
                ]
            )
            logging.info("Saving preprocessor object")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_file_path,test_file_path):
        
        try:
            df_train = pd.read_csv(train_file_path)
            df_test = pd.read_csv(test_file_path)   
            logging.info('Read train and test data')
            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_object(self.data_transformation_config.numerical_features, self.data_transformation_config.categorical_features)
            
            input_features_train = df_train.drop(columns =[self.data_transformation_config.target_column_name])
            target_feature_train = df_train[self.data_transformation_config.target_column_name]
            input_feature_test = df_test.drop(columns =[self.data_transformation_config.target_column_name])
            target_feature_test = df_test[self.data_transformation_config.target_column_name]
            logging.info('Applying preprocessing object on training and testing data')
            
            input_features_trained = preprocessing_obj.fit_transform(input_features_train)
            input_features_tested = preprocessing_obj.transform(input_feature_test)
            train_arr = np.concatenate([input_features_trained, target_feature_train.to_numpy().reshape(-1, 1)], axis=1)
            test_arr = np.concatenate([input_features_tested, target_feature_test.to_numpy().reshape(-1, 1)], axis=1)
            logging.info("Saving processed objects")
            pd.DataFrame(train_arr).to_csv(self.data_transformation_config.transformed_train_file_path, index=False, header=True)
            pd.DataFrame(test_arr).to_csv(self.data_transformation_config.transformed_test_file_path, index=False, header=True)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
           
        except Exception as e:
            raise CustomException(e, sys)
        