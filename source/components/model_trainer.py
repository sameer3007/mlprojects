import os
import sys 
from dataclasses import dataclass
from source.exception import CustomException
from source.logger import logging
from source.utils import save_object
from source.utils import evaluate_models


from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('split trainning and test input data')
            x_train,y_train,x_test,y_test=(train_array[:,:-1],
                                           train_array[:,-1],
                                           test_array[:,:-1],
                                           test_array[:,-1])
            
            models={'RandomForestRegressor':RandomForestRegressor(),
                    'DecisionTreeRegressor':DecisionTreeRegressor(),
                    'GradientBoostingRegressor':GradientBoostingRegressor(),
                    'LinearRegression':LinearRegression(),
                    'KNeighborsRegressor':KNeighborsRegressor(),
                    'XGBRegressor':XGBRegressor(),
                    'AdaBoostRegressor':AdaBoostRegressor(),
                    'SVR':SVR()}
            model_report:dict = evaluate_models(x_train=x_train,
                                                y_train=y_train,
                                                x_test=x_test,
                                                y_test=y_test,
                                                models=models)
            #to get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            #to get best model name from dict
            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('no best model found')
            
            logging.info('best found model on both training and testing dataset')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            predicted=best_model.predict(x_test)
            r2score=r2_score(y_test,predicted)

            return r2score

        except Exception as e:
            raise CustomException(e,sys)

