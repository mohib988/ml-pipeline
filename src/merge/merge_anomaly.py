import pandas as pd
import logging
from zenml import step

@step
def merge_anomaly(df1: pd.DataFrame,df2:pd.DataFrame,df3:pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Merging the anomalies")
        df1['anomaly'] = df1['anomaly'] + df2['anomaly']+df3["anomaly"]
        df1.loc[df1.rate_value<12,"anomaly"]=1
        df1.loc[df1.rate_value>14,"anomaly"]=1
        df1.loc[df1.sales_value<10,"anomaly"]=1
        df1.loc[df1["anomaly"]>0,"anomaly"]=1
        df1.drop(["created_date_time"],axis=1,inplace=True)
        return df1
    except Exception as e:
        logging.error("Error in merging the anomalies")
        raise e