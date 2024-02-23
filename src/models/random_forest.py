from sklearn.model_selection import train_test_split
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
class RandomForestModel:
    def train_random_forest(self,X)->RandomForestClassifier:
        model=pickle.load()
        # Create the training and testing sets

        # Fit a random forest classifier model to our data
        try:
            
            predict=model.predict(X)
            return predict
        except Exception as e:
            logging.error("error in Random Forest",e)
            raise e

    
        # Obtain model predictions
    