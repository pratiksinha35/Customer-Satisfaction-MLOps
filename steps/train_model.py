import logging

import pandas as pd
import mlflow # type: ignore
from zenml import step # type: ignore
from zenml.client import Client # type: ignore


from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin

from .config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
        X_train: pd.DataFrame,  
        y_train: pd.Series,  
        config: ModelNameConfig
    ) -> RegressorMixin:
    '''
    Trains the model on the cleaned data
    
    Args:
        X_train: training data
        y_train: training labels
    '''
    try:
        model = None
        if config.model_name == 'LinearRegression':
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f'Model {config.model_name} not supported')
    except Exception as e:
        logging.error(f'Error in training model: {e}')
        raise e