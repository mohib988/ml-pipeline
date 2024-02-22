from pipelines.training_pipeline import training_pipeline
import mlflow
from zenml.client import Client

def main():
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    return training_pipeline(path='data/AnomalyDetection.csv')

if __name__ == '__main__':
    main()
    # Run the training pipeline
 