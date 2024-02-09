from prophet import Prophet
import pandas as pd
import numpy as np

class AnomalyDetector:
    def __init__(self, att="ntn", seasonality="hourly", frequency="H",df:pd.DataFrame):
        self.att = att
        self.seasonality = seasonality
        self.frequency = frequency
        self.df = df

    def detect_anomalies_prophet(self):
        '''
        Use to detect sales anomalies using Prophet.
        Parameters:
            - df: the dataset
        Returns:
            - casting: DataFrame containing the anomaly detection results
        '''
        # Prepare df in Prophet's required format
        df2=self.df.copy()
        prophet_data = df2.rename(columns={'created_date_time': 'ds', 'sales_value': 'y'})
        prophet_data = prophet_data[[self.att, 'ds', 'y']]
        prophet_data.set_index("ds", inplace=True)

        casting = pd.DataFrame()

        # Loop through each unique restaurant ID
        for restaurant_id in prophet_data[self.att].unique():
            # Subset data for the specific restaurant ID
            subset_data = prophet_data[prophet_data[self.att] == restaurant_id].reset_index()

            # Initialize and fit Prophet model for the specific restaurant ID
            model = Prophet(changepoint_range=0.8, changepoint_prior_scale=0.05)
            model.add_seasonality(name=self.seasonality, period=0.04, fourier_order=20)
            model.add_country_holidays(country_name='Pakistan')
            model.fit(subset_data)

            # Make predictions for the specific restaurant
            future = model.make_future_dataframe(periods=48, freq=self.frequency)
            forecast = model.predict(future)
            forecast_df = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]

            # Detect anomalies based on forecast for this restaurant
            forecasting_final = pd.merge(forecast_df, subset_data, how='inner', left_on='ds', right_on='ds')

            # Calculate the prediction error and uncertainty
            forecasting_final['error'] = forecasting_final['y'] - forecasting_final['yhat']
            forecasting_final['uncertainty'] = forecasting_final['yhat_upper'] - forecasting_final['yhat_lower']

            # Use a factor to identify the outlier or anomaly
            factor = 1.5
            forecasting_final['anomalyfb'] = forecasting_final.apply(lambda x: 1 if (np.abs(x['error']) > factor * x['uncertainty']) else 0, axis=1)

            # Append the results for each restaurant to the casting DataFrame
            casting = casting.append(forecasting_final, ignore_index=True)

        return casting.rename(columns={'ds': 'created_date_time', 'y': 'sales_value'})
    def merge_data( casting:pd. DataFrame)->pd. DataFrame:
        '''     
            Use to merge the anomaly detection results with the original dataset.
        Parameters:
            - df: the original dataset
            - anomaly: the anomaly detection results
        Returns:
            - df: the original dataset with the anomaly detection results
        '''
        df3_with_anomaly = pd.merge(df3, casting[['created_date_time', 'anomalyfb']], on='created_date_time', how='left')
        df3_with_anomaly["anomalyfb"]=df3_with_anomaly["anomalyfb"].fillna(0)
        return df3_with_anomaly
# Example usage:
# anomaly_detector = AnomalyDetector()
# anomalies_result = anomaly_detector.detect_anomalies_prophet(your_dataframe)