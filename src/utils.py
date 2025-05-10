import numpy as np
import os
import pandas as pd
from src.exception import CustomException
import pickle
import dill
from src.logger import logging
import sys
from collections import defaultdict
from time import time


from sklearn.model_selection import cross_val_score



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.
    """
    try:
        logging.info(f"Saving object to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def model_evaluation(models: dict,X_train,y_train,X_test = 0,y_test =0):
    """Return dataframe of model score"""
    results = results = defaultdict(lambda: defaultdict(dict))
    try:
        for model_name,model in models.items():
            st = time()
            cvs = cross_val_score(model,
                                  X_train,
                                  y_train,
                                  cv= 5,
                                  scoring='neg_mean_squared_error')
            train_time = st - time()                                 
            results[model_name]["Mean Squared Error"] = np.mean(-cvs)
            results[model_name]["Training Time"] = -train_time
        return pd.DataFrame(results).stack().reset_index().pivot(
            index="level_1",columns="level_0",values=0).rename_axis(index=[None],columns=[None])
    except Exception as e:
        raise CustomException(e,sys)

