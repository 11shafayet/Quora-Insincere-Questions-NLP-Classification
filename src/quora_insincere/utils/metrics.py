import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(y_true, y_proba, low: float = 0.1, high: float = 0.9, step: float = 0.01):
    thresholds = np.arange(low, high, step)
    best_thresh = 0.5
    best_f1 = 0.0

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return best_thresh, best_f1

