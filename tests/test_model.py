import numpy as np

from model import (
    train_and_predict,
    get_accuracy,
    get_expected_test_size,
    EXPECTED_CLASSES,
)

# Test 1: Sprawdza, czy otrzymujemy jakakolwiek predykcje
def test_predictions_not_none():
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

# Test 2: Sprawdza, czy dlugosc listy predykcji jest wieksza od 0 i czy odpowiada przewidywanej liczbie probek testowych
def test_predictions_length():
    preds, y_test = train_and_predict()

    # Dlugosc musi byc wieksza od zera
    assert len(preds) > 0, "Predictions list should not be empty."

    # Dlugosc predykcji musi odpowiadac liczbie probek testowych
    expected = get_expected_test_size()
    assert len(preds) == expected, (
        f"Expected {expected} predictions, got {len(preds)}."
    )

    # Dodatkowo - dlugosci preds i y_test powinny sie zgadzac
    assert len(preds) == len(y_test), (
        "Predictions and ground-truth length mismatch."
    )

# Test 3: Sprawdza, czy wartosci w predykcjach mieszcza sie w spodziewanym zakresie
def test_predictions_value_range():
    preds, _ = train_and_predict()
    preds_arr = np.asarray(preds)

    # Wszystkie predykcje musza miescic sie w zakresie {0, 1}
    assert preds_arr.min() >= 0, "Predictions contain values < 0."
    assert preds_arr.max() <= 1, "Predictions contain values > 1."

    # Dodatkowo - wszystkie unikalne wartosci musza nalezec do oczekiwanego zbioru klas modelu
    unique_classes = set(np.unique(preds_arr).tolist())
    assert unique_classes.issubset(EXPECTED_CLASSES), (
        f"Predictions contain unexpected classes: {unique_classes}"
    )

# Test 4: Sprawdza, czy model osiaga co najmniej 70% dokladnosci
def test_model_accuracy():
    preds, y_test = train_and_predict()
    acc = get_accuracy(preds, y_test)

    assert acc >= 0.70, (
        f"Model accuracy too low: {acc:.4f} (expected >= 0.70)."
    )
