import pandas as pd
from zenml import step
import logging

class Ingest:
    def __init__(self,path:str):
        self.path=path
    def get_data(self)->pd.DataFrame:
        return pd.read_csv(self.path)
@step
def ingest_data(path:str)->pd.DataFrame:
    try:
        logging.info("getting the data from path",path)
        return Ingest(path).get_data()
    except Exception as e:
        logging.error("Error in getting the data from path",path)
        logging.error(e)
        raise e