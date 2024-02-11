import logging
import pandas as pd
from zenml import step
from src.models.random_forest import RandomForestModel 
from sklearn.ensemble import RandomForestClassifier


@step
def train_model(X_train: pd.DataFrame,y_train:pd.Series) -> RandomForestClassifier:
    '''Train the model
    Args:
        X_train: pd.DataFrame: Training data
        y_train: pd.Series: Training labels
    Returns:
        RandomForestClassifier: Trained model
    '''
    model=RandomForestModel()
    model =model.train_random_forest(X_train, y_train)
    logging.info("Training the model")
    return model
