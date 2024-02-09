import logging
import pandas as pd
from zenml import step


@step
def evaluate_model(df: pd.DataFrame) -> None:
    logging.info("Evaluating the model")
    # Evaluate the model here
    pass