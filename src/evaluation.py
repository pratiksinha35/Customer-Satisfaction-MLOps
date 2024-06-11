import logging

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    '''
    Abstract class defining strategy for evaluating models
    '''
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        '''
        Calculates the scores for the model
        
        Args:
            y_true: true labels
            y_pred: predicted labels
            
        Returns:
            None
        '''
        pass
    

class MSE(Evaluation):
    '''
    Evaluation strategy that uses mean squared error
    '''
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating MSE')
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f'MSE: {mse}')
            return mse
        except Exception as e:
            logging.info(f'Error while calculating MSE: {e}')
            raise e
        
class R2(Evaluation):
    '''
    Evaluation strategy that uses r2 score
    '''
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating R2 score')
            r2 = r2_score(y_true, y_pred)
            logging.info(f'R2 score: {r2}')
            return r2
        except Exception as e:
            logging.info(f'Error while calculating R2 score: {e}')
            raise e
        
class RMSE(Evaluation):
    '''
    Evaluation strategy that uses root mean squared error
    '''
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating RMSE')
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f'MSE: {rmse}')
            return rmse
        except Exception as e:
            logging.info(f'Error while calculating MSE: {e}')
            raise e