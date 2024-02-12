from pipelines.training_pipeline import training_pipeline
import mlflow
from zenml.client import Client


if __name__ == '__main__':
    # Run the training pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(path='data/AnomalyDetection.csv')
 