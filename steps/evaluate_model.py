import logging
from typing import Tuple

import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated # type: ignore
from zenml import step # type: ignore

from src.evaluation import MSE, RMSE, R2

@step
def evaluate_model(
        model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[
        Annotated[float, 'r2_score'],
        Annotated[float, 'rmse']
    ]:
    '''
    Evaluates the model on the ingested data
    
    Args:
        df: the ingested data
    '''
    prediction = model.predict(X_test)
    
    mse_class = MSE()
    mse = mse_class.calculate_scores(y_pred=prediction, y_true=y_test)
    
    r2_class = R2()
    r2_score = r2_class.calculate_scores(y_pred=prediction, y_true=y_test)
    
    rmse_class = RMSE()
    rmse = rmse_class.calculate_scores(y_pred=prediction, y_true=y_test)
    
    return r2_score, rmse