import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import logging
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from zenml import step

class ClassificationEvaluator:
    def __init__(self, model:RandomForestClassifier, X_test:pd.DataFrame, y_test:pd.Series):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_predicted = self.model.predict(self.X_test)
    def pr_matrix(self):
        average_precision = average_precision_score(self.y_test, self.y_predicted)
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_predicted)
        logging.info("precision:",precision)
        logging.info("recall:",recall)
        logging.info("average_precision:",average_precision)
        probs = self.model.predict_proba(self.X_test)
        false_positive_rate, true_positive_rate, _ = roc_curve(self.y_test, probs[:, 1])
        roc_auc = roc_auc_score(self.y_test, probs[:, 1])
        logging.info("false_positive_rate:",false_positive_rate)
        logging.info("true_positive_rate:",true_positive_rate)
        logging.info("probs:",probs)
        logging.info("roc_auc:",roc_auc)
    def evaluate_classification(self):
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_predicted)
        probs = self.model.predict_proba(self.X_test)
        auc_roc_score = roc_auc_score(self.y_test, probs[:, 1])

        # logging.info metrics
        logging.info("Accuracy Score:", accuracy)
        logging.info("AUC ROC score:", auc_roc_score)
        logging.info('Classification report:\n', classification_report(self.y_test, self.y_predicted))
        logging.info('Confusion matrix:\n', confusion_matrix(y_true=self.y_test, y_pred=self.y_predicted))


@step
def evaluate(model:RandomForestClassifier, X_test:pd.DataFrame, y_test:pd.Series):
    try:
        classifier = ClassificationEvaluator(model, X_test, y_test)
        classifier.pr_matrix()
        classifier.evaluate_classification()
    except Exception as e:
        logging.error("Error in evaluating the model", e)
        raise e
    
