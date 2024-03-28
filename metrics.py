from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

def get_classification_scores(y_pred, y_true):
    y_pred_ont_hot = np.argmax(y_pred, axis=1)
    y_true_ont_hot = y_true
    return {
        "f1": f1_score(y_true_ont_hot, y_pred_ont_hot),
        "precision": precision_score(y_true_ont_hot, y_pred_ont_hot),
        "recall": recall_score(y_true_ont_hot, y_pred_ont_hot),
        "accuracy": accuracy_score(y_true_ont_hot, y_pred_ont_hot),
    }

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

def accuracy(y_true, y_pred):
    return accuracy_score(y_true == y_pred)

def precision(y_true, y_pred):
    return precision_score(y_true == y_pred)

def recall(y_true, y_pred):
    return recall_score(y_true == y_pred)