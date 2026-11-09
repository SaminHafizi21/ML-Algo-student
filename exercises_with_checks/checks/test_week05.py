"""Checks voor het week05 checkpoint (Herhalingslabo Classificatie).

Elke test valideert exact één verplichte variabele uit het notebook:
bestaat -> type -> drempelwaarde. Geen implementatiehints in de meldingen.

Let op: de creditcard-oefening zit bewust NIET in de autocheck. De dataset is
te groot voor GitHub en wordt via het schoolplatform verdeeld; de checks zouden
anders afhangen van een bestand dat in CI niet aanwezig is.
"""
import numpy as np


def _get(week, name):
    val = week.get(name)
    assert val is not None, (
        f"Variabele '{name}' ontbreekt of heeft geen waarde "
        f"(verwacht: gevuld volgens de opdracht)."
    )
    return val


# ---------------- Diabetes: SVC ----------------

def test_diabetes_svc_y_pred(week05):
    y_pred = _get(week05, "y_pred_diabetes_svc")
    assert len(np.asarray(y_pred)) == 154, (
        f"y_pred_diabetes_svc heeft {len(np.asarray(y_pred))} voorspellingen, verwacht 154"
    )


def test_diabetes_svc_accuracy(week05):
    acc = _get(week05, "accuracy_diabetes_svc")
    assert float(acc) >= 0.70, (
        f"accuracy_diabetes_svc={float(acc):.4f} is te laag (verwacht >= 0.70)"
    )


def test_diabetes_svc_accuracy_boven_gok(week05):
    acc = float(_get(week05, "accuracy_diabetes_svc"))
    assert acc > 0.5, (
        f"accuracy_diabetes_svc={acc:.4f} is niet beter dan gokken (verwacht > 0.5)"
    )


# ---------------- Diabetes: KNN ----------------

def test_diabetes_knn_y_pred(week05):
    y_pred = _get(week05, "y_pred_diabetes_knn")
    assert len(np.asarray(y_pred)) == 154, (
        f"y_pred_diabetes_knn heeft {len(np.asarray(y_pred))} voorspellingen, verwacht 154"
    )


def test_diabetes_knn_accuracy(week05):
    acc = _get(week05, "accuracy_diabetes_knn")
    assert float(acc) >= 0.60, (
        f"accuracy_diabetes_knn={float(acc):.4f} is te laag (verwacht >= 0.60)"
    )


# ---------------- Churn ----------------

def test_churn_y_pred(week05):
    y_pred = _get(week05, "y_pred_churn")
    assert len(np.asarray(y_pred)) == 1409, (
        f"y_pred_churn heeft {len(np.asarray(y_pred))} voorspellingen, verwacht 1409"
    )


def test_churn_accuracy(week05):
    acc = _get(week05, "accuracy_churn")
    assert float(acc) >= 0.75, (
        f"accuracy_churn={float(acc):.4f} is te laag (verwacht >= 0.75)"
    )


# ---------------- Meerkeuzevragen ----------------

def test_antwoord_algoritme(week05):
    antw = _get(week05, "antwoord_algoritme")
    assert str(antw).strip().upper() == "B", (
        f"antwoord_algoritme={str(antw)!r} is niet correct"
    )


def test_antwoord_ongebalanceerd(week05):
    antw = _get(week05, "antwoord_ongebalanceerd")
    assert str(antw).strip().upper() == "B", (
        f"antwoord_ongebalanceerd={str(antw)!r} is niet correct"
    )
