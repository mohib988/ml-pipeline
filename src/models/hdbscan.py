import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from zenml import step

class HDBClustering:
  def __init__(self,df:pd.DataFrame,**kargs):
    self.df=df
    self.eps = 0.8  # Adjust the epsilon value based on your df2_scaled
    self.min_samples = 80
    self.hdbscan = HDBSCAN(cluster_selection_epsilon=0.8, min_cluster_size=20)

# Assuming 'self.df' is your DataFrame with the relevant data
# Feature engineering
  def run_hdb_scan(self,hour:bool=True):
      if(hour):
        self.df['hour'] = self.df['created_date_time'].dt.hour
      self.df['day'] = self.df['created_date_time'].dt.day
      # self.df = self.df.reset_index(drop=True)

      # Select relevant features for prediction
      features = [  'sales_value', 'day', 'month',"location"]
      if (hour):
        features.append("hour")


      # Prepare input data
      X = self.df[features]
      X["sales_value"] = np.log1p(np.abs(X["sales_value"]))
      # One-hot encode the 'location' column
      X=pd.get_dummies(X,columns=['location'])

      # Select relevant features for clustering
      # Standardize the selected features
      scaler = StandardScaler()
      df2_scaled = scaler.fit_transform(X)
      MinPts=5
      # # Apply DBSCAN
      self.df['anomaly'] = self.hdbscan.fit_predict(df2_scaled)
      self.df['anomaly'] =(self.df["anomaly"] == -1).astype(int)*4

      # Filter rows marked as anomalies
      # self.df["sales_value"] = np.expm1(np.abs(self.df["sales_value"]))
      return self.df
@step
def run_hdb_clustering(df:pd.DataFrame,hour:bool=True,**kargs)->pd.DataFrame:
  HDB_clustering=HDBClustering(df)
  df=HDB_clustering.run_hdb_scan(hour)
  return df
