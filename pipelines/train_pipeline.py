from zenml import pipeline # type: ignore

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

@pipeline
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, y_train)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    