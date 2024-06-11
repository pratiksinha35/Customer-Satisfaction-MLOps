import logging

import pandas as pd
from zenml import step # type: ignore

@step
def train_model(df: pd.DataFrame) -> None:
    '''
    Trains the model on the cleaned data
    
    Args:
        df: the cleaned data
    '''
    pass