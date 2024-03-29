from pipelines.training_pipeline import training_pipeline
import json
from typing import List, Dict, Any
import mlflow
import pandas as pd
from zenml.client import Client

def main(df):
    df_dict_list = df.to_dict(orient='records')
    df_json_str = json.dumps(df_dict_list)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    return training_pipeline(df_json_str)

if __name__ == '__main__':
    df=pd.read_csv("./data/AnomalyDetection.csv")
    df_dict_list = df.to_dict(orient='records')
    df_json_str = json.dumps(df_dict_list)
    main(df)
    # Run the training pipeline
 