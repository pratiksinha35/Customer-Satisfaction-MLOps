import numpy as np
import pandas as pd
from zenml import pipeline, step # type: ignore
from zenml.config import DockerSettings # type: ignore
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT # type: ignore
from zenml.integrations.constants import MLFLOW # type: ignore
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer # type: ignore
from zenml.integrations.mlflow.services import MLFlowDeploymentService # type: ignore
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step # type: ignore
from zenml.steps import BaseParameters, Output # type: ignore

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    '''Deployment trigger config'''
    min_accuracy: float = 0.92
    
@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
):
    '''Implements a simple model deployment trigger that looks at the model accuracy'''
    return accuracy >= config.min_accuracy

@pipeline(enable_cache=True, settings={'docker': docker_settings})
def continuous_deployment_pipeline(
    data_path: str = r'D:\Pratik\Customer Satisfaction MLOps\data\olist_customers_dataset.csv',
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, y_train)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )
    
    