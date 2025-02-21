import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from source.exception import CustomException
from source.logger import logging
from source.utils import save_object


@dataclass
class DataTransformConfig:
    transformer_obj_path_file=os.path.join('artifacts','transformer.pkl')

class Datatransform:
    def __init__(self):
        self.data_tranformation_config=DataTransformConfig()

    def get_data_transformer_obj(self):
        ''' this function is responsible for data transformation'''
        try:
            numerical_columns=['reading score', 'writing score']
            categorical_columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            numerical_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                               ('scaler',StandardScaler(with_mean=False))])
            categorical_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                          ('onehotencoder',OneHotEncoder()),
                                          ('scaler',StandardScaler(with_mean=False))])
            logging.info(f'categorical columns: {categorical_columns}')
            logging.info(f'numerical columns: {numerical_columns}')

            transformer=ColumnTransformer([('numerical_pipeline',numerical_pipeline,numerical_columns),
                                          ('categorical_pipeline',categorical_pipeline,categorical_columns)])
            
            return transformer
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_past):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_past)

            logging.info('read train and test data completed')
            logging.info('obtaining transformation object')

            transformation_obj=self.get_data_transformer_obj()

            target_column_name='math score'
            numerical_columns=['reading score', 'writing score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('applying transformation object on trainning and testing dataframes')

            input_feature_train_array=transformation_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=transformation_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_array,np.array(target_feature_test_df)]

            logging.info('saved transformed objects')

            save_object(file_path=self.data_tranformation_config.transformer_obj_path_file,
                        obj=transformation_obj)

            return (train_arr,
                    test_arr,
                    self.data_tranformation_config.transformer_obj_path_file)

        except Exception as e:
            raise CustomException(e,sys)