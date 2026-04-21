import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Dane treningowe
X_DATA = np.array([
    [1, 40],
    [2, 50],
    [3, 60],
    [4, 70],
    [5, 80],
    [6, 90],
], dtype=float)

Y_DATA = np.array([0, 0, 0, 1, 1, 1])


def _build_model():
    model = LogisticRegression()
    model.fit(X_DATA, Y_DATA)
    return model


def train_and_predict():
    model = _build_model()
    predictions = model.predict(X_DATA)
    return predictions, Y_DATA


def get_accuracy(predictions, y_true):
    return accuracy_score(y_true, predictions)


def get_expected_test_size():
    return len(X_DATA)

EXPECTED_CLASSES = {0, 1}
