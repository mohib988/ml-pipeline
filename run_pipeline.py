from pipelines.training_pipeline import training_pipeline
import mlflow
import pandas as pd
from zenml.client import Client

def main(df:pd.DataFrame):
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    return training_pipeline(pd)

if __name__ == '__main__':
    main()
    # Run the training pipeline
 