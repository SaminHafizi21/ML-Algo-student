"""Checkpoint-tests week 2: SVM (opgave-notebook van de student)."""
import numpy as np
import pytest


# ---------- Oefening 1: Breast Cancer ----------
def test_oef1_accuracy_breast_cancer(week02):
    acc = week02.get("accuracy_breast_cancer")
    assert acc is not None, "variabele 'accuracy_breast_cancer' ontbreekt"
    acc = float(acc)
    assert 0.9 <= acc <= 1.0, f"accuracy_breast_cancer={acc} is te laag (verwacht >= 0.9)"


def test_oef1_y_pred_breast_cancer(week02):
    y_pred = week02.get("y_pred_breast_cancer")
    assert y_pred is not None, "variabele 'y_pred_breast_cancer' ontbreekt"
    assert len(np.asarray(y_pred)) == 143, "y_pred_breast_cancer heeft niet de juiste lengte (test_size=0.25 van 569)"


def test_oef1_theorie_scaling(week02):
    antw = week02.get("antwoord_scaling")
    assert antw is not None, "variabele 'antwoord_scaling' ontbreekt"
    assert str(antw).strip().upper() == "B", "Fout: SVM's zijn afstand- en inwendig-productgebaseerd"


# ---------- Oefening 2: Digits ----------
def test_oef2_accuracy_digits(week02):
    acc = week02.get("accuracy_digits")
    assert acc is not None, "variabele 'accuracy_digits' ontbreekt"
    acc = float(acc)
    assert 0.9 <= acc <= 1.0, f"accuracy_digits={acc} is te laag (verwacht >= 0.9)"


def test_oef2_y_pred_digits(week02):
    y_pred = week02.get("y_pred_digits")
    assert y_pred is not None, "variabele 'y_pred_digits' ontbreekt"
    assert len(np.asarray(y_pred)) == 450, "y_pred_digits heeft niet de juiste lengte (test_size=0.25 van 1797)"


def test_oef2_theorie_kernel(week02):
    antw = week02.get("antwoord_kernel")
    assert antw is not None, "variabele 'antwoord_kernel' ontbreekt"
<<<<<<< Updated upstream
    assert str(antw).strip().upper() == "A", ""
=======
    assert str(antw).strip().upper() == "C", "Fout: kijk naar de resultaten van de verschillende kernels"
>>>>>>> Stashed changes


# ---------- Oefening 3: Wine ----------
def test_oef3_accuracy_wine(week02):
    acc = week02.get("accuracy_wine")
    assert acc is not None, "variabele 'accuracy_wine' ontbreekt"
    acc = float(acc)
    assert 0.85 <= acc <= 1.0, f"accuracy_wine={acc} is te laag (verwacht >= 0.85)"


# ---------- Oefening 4: SVR ----------
def test_oef4_y_pred_svr(week02):
    y_pred = week02.get("y_pred_svr")
    assert y_pred is not None, "variabele 'y_pred_svr' ontbreekt"
    assert len(np.asarray(y_pred)) == 50, "y_pred_svr heeft niet de juiste lengte (50 trainingspunten)"


def test_oef4_mse_svr(week02):
    mse = week02.get("mse_svr")
    assert mse is not None, "variabele 'mse_svr' ontbreekt"
    mse = float(mse)
    assert 0 < mse < 0.05, f"mse_svr={mse} valt buiten het verwachte bereik (0 < mse < 0.05)"