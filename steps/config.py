from zenml.steps import BaseParameters # type: ignore

class ModelNameConfig(BaseParameters):
    '''
    Model Config
    '''
    model_name: str = 'LinearRegression'