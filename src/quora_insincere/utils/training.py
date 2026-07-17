import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping


def make_class_weight_dict(y_train) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


def make_early_stopping():
    return EarlyStopping(
        monitor="val_auc",
        patience=3,
        mode="max",
        restore_best_weights=True,
    )

