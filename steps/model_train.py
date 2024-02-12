import logging
import pandas as pd
from zenml import step
from src.models.random_forest import RandomForestModel 
from sklearn.base import ClassifierMixin
import mlflow
from zenml.client import Client

experiment_tracker=Client().active_stack.experiment_tracker 


@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame,y_train:pd.Series) -> ClassifierMixin:
    """Train the model
    Args:
        X_train: pd.DataFrame: Training data
        y_train: pd.Series: Training labels
    Returns:
        ClassifierMixin: Trained model
    """
    mlflow.sklearn.autolog()
    model=RandomForestModel()
    model =model.train_random_forest(X_train, y_train)
    logging.info("Training the model")
    return model
