import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaner,DataPreprocessingStrategy
from typing import Union
@step
def clean_data(df: pd.DataFrame) :
    try:
        process_strategy=DataPreprocessingStrategy()
        data_cleaner=DataCleaner(process_strategy,df)
        processed_data=data_cleaner.handle_data()
        logging.info("Data Preprocessing Done")
        return processed_data
        # divide_strategy=DataDivideStrategy()
        # data_cleaner=DataCleaner(divide_strategy,df)
        # X_train, X_test, y_train, y_test=data_cleaner.handle_data()
    except Exception as e:
        logging.error("Error in Data Preprocessing", e)

    # logging.info("Cleaning the data")
    # # Drop rows with missing values
    # df = df.dropna()
    # return df