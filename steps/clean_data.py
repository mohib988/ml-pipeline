import logging
import pandas as pd
from zenml import step


@step
def clean_data(df: pd.DataFrame) -> None:
    # logging.info("Cleaning the data")
    # # Drop rows with missing values
    # df = df.dropna()
    # return df
    pass