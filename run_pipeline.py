from zenml.client import Client #type: ignore

from pipelines.train_pipeline import train_pipeline

if __name__ == '__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=r'D:\Pratik\Customer Satisfaction MLOps\data\olist_customers_dataset.csv')
