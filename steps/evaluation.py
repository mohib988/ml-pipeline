import matplotlib.pyplot as plt
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
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_predicted = self.model.predict(self.X_test)
    def plot_pr_curve(self):
        average_precision = average_precision_score(self.y_test, self.y_predicted)
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_predicted)
        print("precision:",precision)
        print("recall:",recall)
        print("average_precision:",average_precision)
        probs = self.model.predict_proba(self.X_test)
        false_positive_rate, true_positive_rate, _ = roc_curve(self.y_test, probs[:, 1])
        roc_auc = roc_auc_score(self.y_test, probs[:, 1])
        print("false_positive_rate:",false_positive_rate)
        print("true_positive_rate:",true_positive_rate)
        print("probs:",probs)
        print("roc_auc:",roc_auc)
    def evaluate_classification(self):
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_predicted)
        probs = self.model.predict_proba(self.X_test)
        auc_roc_score = roc_auc_score(self.y_test, probs[:, 1])

        # Print metrics
        print("Accuracy Score:", accuracy)
        print("AUC ROC score:", auc_roc_score)
        print('Classification report:\n', classification_report(self.y_test, self.y_predicted))
        print('Confusion matrix:\n', confusion_matrix(y_true=self.y_test, y_pred=self.y_predicted))


@step
def evaluate(model, X_test, y_test):
    classifier = ClassificationEvaluator(model, X_test, y_test)
    classifier.plot_pr_curve()
    classifier.plot_roc_curve()
    classifier.evaluate_classification()
