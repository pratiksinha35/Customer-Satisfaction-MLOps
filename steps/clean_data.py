import logging

import pandas as pd
from zenml import step # type: ignore

@step
def clean_df(df: pd.DataFrame) -> None:
    '''
    Cleans the ingested data
    
    Args:
        df: the ingested data
    '''
    pass