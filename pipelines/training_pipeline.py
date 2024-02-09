from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model
@pipeline(enable_cache=True)
def training_pipeline(path:str):
    # Define pipeline here
    df=ingest_data(path)
    # X=clean_data(df)
    df=clean_data(df)
    train_model(df)
    evaluate_model(df)
