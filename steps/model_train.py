import logging
import pandas as pd
from zenml import step


@step
def train_model(df: pd.DataFrame) -> None:
    logging.info("Training the model")
    # Train the model here
    pass