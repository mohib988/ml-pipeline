from sklearn.neighbors import LocalOutlierFactor
from pandas import DataFrame

class LocalOutlierDetection:
    def __init__(self,**kargs):
        self.contamination = contamination
        self.neighbors = neighbors
        self.clf = LocalOutlierFactor(**kargs)

    def fit_predict(self, df: DataFrame) -> DataFrame:
        y_pred = self.clf.fit_predict(df)
        df["anomaly_loc"] = (y_pred == -1).astype(int)
        return df