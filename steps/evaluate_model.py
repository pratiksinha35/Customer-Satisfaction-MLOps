import logging

import pandas as pd
from zenml import step # type: ignore

@step
def evaluate_model(df: pd.DataFrame) -> None:
    '''
    Evaluates the model on the ingested data
    
    Args:
        df: the ingested data
    '''
    pass