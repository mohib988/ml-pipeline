import pandas as pd

from typing import Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame

class IsolationForest:
    def __init__(self, **kargs):
        """
        Initialize the IsolationForest class.

        Parameters:
        - contamination: float, the proportion of outliers in the data.
        """
        self.contamination = contamination
        self.isolation_forest = IsolationForest(**kargs)

    def apply_isolation_forest(self, df: DataFrame) -> DataFrame:
        """
        Apply Isolation Forest to detect outliers/anomalies in a DataFrame.

        Parameters:
        - df: DataFrame, the input DataFrame containing numeric columns.

        Returns:
        - DataFrame, a copy of the input DataFrame with an additional column 'anomaly'
          indicating whether each row is an outlier (1) or not (0).
        """
        # Selecting only numeric columns
        df_numeric = df.select_dtypes(include=['float64', 'int64']).copy()

        # Handling missing values (filling with mean for example)
        # Normalize the data if needed (using StandardScaler)
        # scaler = StandardScaler()
        # df_normalized = scaler.fit_transform(df_numeric)

        # Applying Isolation Forest
        outliers = self.isolation_forest.fit_predict(df_numeric)

        # Add 'anomaly' column to the original DataFrame
        df['anomaly_if'] = (outliers == -1).astype(int)

        return df

# Example usage:
# Initialize IsolationForest object




