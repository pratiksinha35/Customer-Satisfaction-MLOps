import logging

import pandas as pd
from zenml import step # type: ignore

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin

from .config import ModelNameConfig

@step
def train_model(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series,  
        y_test: pd.Series,
        config: ModelNameConfig
    ) -> RegressorMixin:
    '''
    Trains the model on the cleaned data
    
    Args:
        X_train: training data
        y_train: training labels
        X_test: testing data
        y_test: testing labels
    '''
    try:
        model = None
        if config.model_name == 'LinearRegression':
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f'Model {config.model_name} not supported')
    except Exception as e:
        logging.error(f'Error in training model: {e}')
        raise e