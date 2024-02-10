import pandas as pd
from zenml import step
from typing import Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame

class Isolation_Forest:
    def __init__(self):
        """
        Initialize the IsolationForest class.

        Parameters:
        - contamination: float, the proportion of outliers in the data.
        """
        self.isolation_forest = IsolationForest()
    def apply_isolation_forest(self,df) -> pd.DataFrame:
        """
        Parameters:
        - df: DataFrame, the input DataFrame containing numeric columns.

        Returns:
        - DataFrame
        """
        # Selecting only numeric columns
        df_numeric =df.select_dtypes(include=['float64', 'int64'])
        # Handling missing values (filling with mean for example)
        # Normalize the data if needed (using StandardScaler)
        # scaler = StandardScaler()
        # df_normalized = scaler.fit_transform(df_numeric)

        # Applying Isolation Forest
        outliers = self.isolation_forest.fit_predict(df_numeric)

        # Add 'anomaly' column to the original DataFrame
        df['anomaly'] = (outliers == -1).astype(int)

        return df

# Example usage:
# Initialize IsolationForest object
@step
def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    isolationForest = Isolation_Forest()
    # Apply Isolation Forest to the input DataFrame
    df = isolationForest.apply_isolation_forest(df)
    return df



