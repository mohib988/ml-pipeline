from zenml import pipeline
import pandas as pd
import json
import numpy as np
from steps.ingest_data import ingest_data,make_data
from steps.clean_data import clean_data,split_data
from steps.model_train import train_model
from steps.evaluation import evaluate
from src.models.isolation_forest import run_isolation_forest    
from typing import List, Dict, Any
from src.merge.merge_anomaly import merge_anomaly
from src.models.loc import run_loc
from src.models.hdbscan import run_hdb_clustering
# from src.models.fbprophet import run_prophet
@pipeline(enable_cache=False)
def training_pipeline(df1):
    # Define pipeline here
    # df=ingest_data(path)
    df=make_data(df1)
    df=clean_data(df)
    
    anomaly1=run_isolation_forest(df)
    anomaly2=run_hdb_clustering(df)
    anomaly3=train_model(df)
    merge_outliers=merge_anomaly(anomaly1,anomaly2,anomaly3)

    return merge_outliers
