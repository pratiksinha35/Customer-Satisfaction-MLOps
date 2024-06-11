import logging

from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin

class Model(ABC):
    '''
    Abstract class for all model
    '''
    
    @abstractmethod
    def train(self, X_train, y_train):
        '''
        Trains the model 
        
        Args:
            X_train: training data
            y_train: training labels
        
        Returns:
            None
        '''
        pass
    
class LinearRegressionModel(Model):
    '''
    Linear Regression Model
    '''
    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:
        '''
        Trains the model 
        
        Args:
            X_train: training data
            y_train: training labels
        
        Returns:
            Linear Regression Model
        '''
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info('Model training complete')
            return reg
        except Exception as e:
            logging.error(f'Error while model training: {e}')
            raise e
        