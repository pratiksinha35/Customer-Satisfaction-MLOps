from zenml import pipeline # type: ignore

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

@pipeline
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    clean_df(df)
    train_model(df)
    evaluate_model(df)
    