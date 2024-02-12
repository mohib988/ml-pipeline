from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data,split_data
from steps.model_train import train_model
from steps.evaluation import evaluate
from src.models.isolation_forest import run_isolation_forest    
from src.merge.merge_anomaly import merge_anomaly
from src.models.loc import run_loc
from src.models.fbprophet import run_prophet
@pipeline(enable_cache=True)
def training_pipeline(path:str):
    # Define pipeline here
    df=ingest_data(path)
    # X=clean_data(df)
    df=clean_data(df)
    
    df1=run_isolation_forest(df)
    df2=run_loc(df)
    df3=run_prophet(df)
    df_merge=merge_anomaly(df1,df2,df3)
    X_train, X_test, y_train, y_test=split_data(df_merge)
    # localOutlierDetection.fit_predict(df)

    model=train_model(X_train, y_train)
    a=evaluate(model, X_test, y_test)
