import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import load_object
@dataclass
class PredictorConfig:
    model_path = os.path.join("artifacts",'model.pkl')
    preprocessor_path = os.path.join("artifacts",'preprocessor.pkl')
@dataclass
class CustomData:
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float

    def get_data_as_dataframe(self):
        try:
            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e)
    
class Predictor:
    def __init__(self):
        self.predictor_config = PredictorConfig()
        self.model = load_object(self.predictor_config.model_path)
        self.preprocessor = load_object(self.predictor_config.preprocessor_path)
    def predict(self,data:CustomData):
        try:
            
            input_df = data.get_data_as_dataframe()
            
            transformed_df = self.preprocessor.transform(input_df)

            prediction = self.model.predict(transformed_df)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    predictor = Predictor()
    data = CustomData(gender="female",
                      race_ethnicity="group A",
                      parental_level_of_education= "bachelor's degree",
                      lunch= "standard",
                      test_preparation_course="none",
                      reading_score= 70,
                      writing_score= 80 )
    print(predictor.predict(data))