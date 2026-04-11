import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score


def apply_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    adjusted = probabilities - thresholds.reshape(1, -1)
    return np.argmax(adjusted, axis=1)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
        "per_class_recall": recall_score(
            y_true, y_pred, average=None, labels=list(range(len(class_names)))
        ).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }
