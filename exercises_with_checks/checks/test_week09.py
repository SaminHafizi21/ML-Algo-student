"""Checks voor het week09 checkpoint (anomaliedetectie & data drift).

Elke test valideert exact één verplichte variabele of klasse uit het notebook:
bestaat -> type -> drempelwaarde. Geen implementatiehints in de meldingen.
"""
import numpy as np
import pandas as pd


def _get(week, name):
    val = week.get(name)
    assert val is not None, (
        f"Variabele '{name}' ontbreekt of heeft geen waarde "
        f"(verwacht: gevuld volgens de opdracht)."
    )
    return val


# ---------------- Oefening 1: anomaliedetectie ----------------


def test_data_is_dataframe(week09):
    data = _get(week09, "data")
    assert isinstance(data, pd.DataFrame), (
        f"data is van type {type(data).__name__}, verwacht een pandas DataFrame"
    )


def test_data_shape(week09):
    data = _get(week09, "data")
    assert data.shape[1] == 3, (
        f"data heeft {data.shape[1]} kolommen, verwacht 3 (Duration, SrcBytes, DstBytes)"
    )
    assert data.shape[0] >= 90, (
        f"data bevat maar {data.shape[0]} rijen, verwacht minstens 90 observaties"
    )


def test_y_pred_anomaly_length(week09):
    y_pred = _get(week09, "y_pred_anomaly")
    n = week09.get("data").shape[0]
    preds = np.asarray(y_pred)
    assert len(preds) == n, (
        f"y_pred_anomaly heeft {len(preds)} waarden, verwacht {n} "
        f"(één per observatie in de data)"
    )


def test_y_pred_anomaly_labels(week09):
    y_pred = _get(week09, "y_pred_anomaly")
    labels = set(np.asarray(y_pred).tolist())
    assert labels <= {1, -1}, (
        f"y_pred_anomaly bevat labels {labels}; verwacht enkel 1 (normaal) en -1 (anomalie)"
    )
    assert -1 in labels, "y_pred_anomaly bevat geen enkele anomalie (label -1)"


def test_n_anomalies_if(week09):
    n = _get(week09, "n_anomalies_if")
    assert isinstance(n, (int, np.integer)) or hasattr(n, "item"), (
        f"n_anomalies_if is van type {type(n).__name__}, verwacht een getal"
    )
    n = int(n)
    assert n == 7, (
        f"n_anomalies_if={n}, verwacht 7 (bij contamination=0.07)"
    )


# ---------------- Oefening 2: feature drift met KS-test ----------------


def test_p_value_feature_bestaat(week09):
    p = _get(week09, "p_value_feature")
    assert isinstance(p, (int, float)) or hasattr(p, "item"), (
        f"p_value_feature is van type {type(p).__name__}, verwacht een p-waarde"
    )


def test_p_value_feature_geldig(week09):
    p = float(week09["p_value_feature"])
    assert 0.0 <= p <= 1.0, (
        f"p_value_feature={p} ligt niet tussen 0 en 1 (verwacht een geldige p-waarde)"
    )


def test_data_drift_detected_is_boolean(week09):
    d = _get(week09, "data_drift_detected")
    assert isinstance(d, (bool, np.bool_)) or d in (True, False), (
        f"data_drift_detected is van type {type(d).__name__}, verwacht een boolean"
    )


def test_data_drift_detected_consistent(week09):
    """data_drift_detected moet True zijn precies wanneer p_value_feature < 0.05."""
    p = float(week09["p_value_feature"])
    d = bool(week09["data_drift_detected"])
    assert d == (p < 0.05), (
        f"data_drift_detected={d} is niet consistent met p_value_feature={p:.4f} "
        f"(verwacht {p < 0.05})"
    )


# ---------------- Oefening 3: DriftDetector klasse ----------------


def test_drift_detector_klasse(week09):
    D = _get(week09, "DriftDetector")
    assert isinstance(D, type), (
        f"DriftDetector is {type(D).__name__}, verwacht een klasse"
    )
    inst = D()
    assert hasattr(inst, "fit") and hasattr(inst, "detect_drift"), (
        "DriftDetector mist de methoden 'fit' en 'detect_drift'"
    )


def test_drift_detector_works(week09):
    D = _get(week09, "DriftDetector")
    np.random.seed(1)
    ref = np.random.normal(0, 1, 5000)
    shifted = np.random.normal(3.0, 1, 5000)
    d = D(significance_level=0.05)
    d.fit(ref)
    result = d.detect_drift(shifted)
    assert bool(result) is True, (
        f"detect_drift geeft {result} terug op een duidelijke verschuiving, verwacht True"
    )


def test_drift_detector_geen_drift(week09):
    D = _get(week09, "DriftDetector")
    np.random.seed(2)
    ref = np.random.normal(0, 1, 5000)
    same = np.random.normal(0, 1, 5000)
    d = D(significance_level=0.05)
    d.fit(ref)
    result = d.detect_drift(same)
    assert bool(result) is False, (
        f"detect_drift geeft {result} terug op onveranderde data, verwacht False"
    )


def test_drift_detected_vb(week09):
    dvb = _get(week09, "drift_detected_vb")
    assert bool(dvb) is True, (
        f"drift_detected_vb={dvb}, verwacht True (duidelijke verschuiving in het voorbeeld)"
    )


# ---------------- Meerkeuzevraag ----------------


def test_antwoord_drift(week09):
    antw = _get(week09, "antwoord_drift")
    assert str(antw).strip().upper() == "A", (
        f"antwoord_drift={str(antw)!r} is niet correct"
    )