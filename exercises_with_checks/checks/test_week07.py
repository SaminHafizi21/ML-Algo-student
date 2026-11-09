"""Checks voor het week07 checkpoint (dimensie reductie met PCA).

Elke test valideert exact één verplichte variabele uit het notebook:
bestaat -> type -> drempelwaarde. Geen implementatiehints in de meldingen.
"""
import numpy as np


def _get(week, name):
    val = week.get(name)
    assert val is not None, (
        f"Variabele '{name}' ontbreekt of heeft geen waarde "
        f"(verwacht: gevuld volgens de opdracht)."
    )
    return val


# ---------------- Oefening 1: SECOM (590 features) ----------------

def test_n_components_secom(week07):
    n = _get(week07, "n_components_secom")
    assert isinstance(n, (int, np.integer)) and 150 <= int(n) <= 190, (
        f"n_components_secom={n!r} is geen geldig aantal PCA-componenten "
        f"voor 95% variantie (verwacht 150 t/m 190)"
    )


def test_reduction_pct_secom(week07):
    pct = _get(week07, "reduction_pct_secom")
    assert 64.0 <= float(pct) <= 78.0, (
        f"reduction_pct_secom={float(pct):.2f} is geen realistische "
        f"datareductie bij 95% variantie (verwacht 64% t/m 78%)"
    )


def test_X_pca_secom_shape(week07):
    X = np.asarray(_get(week07, "X_pca_secom"))
    n = int(_get(week07, "n_components_secom"))
    assert X.ndim == 2 and X.shape == (1567, n), (
        f"X_pca_secom heeft vorm {X.shape}, verwacht (1567, {n}) "
        f"(zelfde aantal rijen als de data, {n} componenten)"
    )


# ---------------- Oefening 2: Steel Plates (27 features) ----------------

def test_n_components_steel(week07):
    n = _get(week07, "n_components_steel")
    assert isinstance(n, (int, np.integer)) and 11 <= int(n) <= 15, (
        f"n_components_steel={n!r} is geen geldig aantal PCA-componenten "
        f"voor 95% variantie (verwacht 11 t/m 15)"
    )


def test_reduction_pct_steel(week07):
    pct = _get(week07, "reduction_pct_steel")
    assert 42.0 <= float(pct) <= 62.0, (
        f"reduction_pct_steel={float(pct):.2f} is geen realistische "
        f"datareductie bij 95% variantie (verwacht 42% t/m 62%)"
    )


def test_X_pca_steel_shape(week07):
    X = np.asarray(_get(week07, "X_pca_steel"))
    n = int(_get(week07, "n_components_steel"))
    assert X.ndim == 2 and X.shape == (1941, n), (
        f"X_pca_steel heeft vorm {X.shape}, verwacht (1941, {n}) "
        f"(zelfde aantal rijen als de data, {n} componenten)"
    )


# ---------------- Oefening 3: APS Failure (170 features) ----------------

def test_n_components_aps(week07):
    n = _get(week07, "n_components_aps")
    assert isinstance(n, (int, np.integer)) and 70 <= int(n) <= 95, (
        f"n_components_aps={n!r} is geen geldig aantal PCA-componenten "
        f"voor 95% variantie (verwacht 70 t/m 95)"
    )


def test_reduction_pct_aps(week07):
    pct = _get(week07, "reduction_pct_aps")
    assert 42.0 <= float(pct) <= 60.0, (
        f"reduction_pct_aps={float(pct):.2f} is geen realistische "
        f"datareductie bij 95% variantie (verwacht 42% t/m 60%)"
    )


def test_X_pca_aps_shape(week07):
    X = np.asarray(_get(week07, "X_pca_aps"))
    n = int(_get(week07, "n_components_aps"))
    assert X.ndim == 2 and X.shape == (76000, n), (
        f"X_pca_aps heeft vorm {X.shape}, verwacht (76000, {n}) "
        f"(train + test gecombineerd, {n} componenten)"
    )


# ---------------- Meerkeuzevragen ----------------

def test_antwoord_scaling(week07):
    antw = _get(week07, "antwoord_scaling")
    assert str(antw).strip().upper() == "B", (
        f"antwoord_scaling={str(antw)!r} is niet correct"
    )


def test_antwoord_95(week07):
    antw = _get(week07, "antwoord_95")
    assert str(antw).strip().upper() == "C", (
        f"antwoord_95={str(antw)!r} is niet correct"
    )
