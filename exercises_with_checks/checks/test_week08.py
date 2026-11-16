"""Checks voor het week08 checkpoint (recommender systems met ALS).

Elke test valideert exact één verplichte variabele uit het notebook:
bestaat -> type -> drempelwaarde. Geen implementatiehints in de meldingen.
"""
import pandas as pd
import scipy.sparse as sp


def _get(week, name):
    val = week.get(name)
    assert val is not None, (
        f"Variabele '{name}' ontbreekt of heeft geen waarde "
        f"(verwacht: gevuld volgens de opdracht)."
    )
    return val


# ---------------- Oefening 1: speelgoed-data ----------------


def test_user_item_matrix_is_dataframe(week08):
    m = _get(week08, "user_item_matrix")
    assert isinstance(m, pd.DataFrame), (
        f"user_item_matrix is van type {type(m).__name__}, verwacht een pandas DataFrame"
    )


def test_user_item_matrix_shape(week08):
    m = _get(week08, "user_item_matrix")
    assert m.shape == (6, 6), (
        f"user_item_matrix heeft vorm {m.shape}, verwacht (6, 6) "
        f"(6 users, 6 songs)"
    )


def test_sparse_user_item_matrix_type(week08):
    m = _get(week08, "sparse_user_item_matrix")
    assert sp.isspmatrix_csr(m), (
        f"sparse_user_item_matrix is van type {type(m).__name__}, "
        f"verwacht een scipy CSR-matrix"
    )


def test_sparse_user_item_matrix_shape(week08):
    m = _get(week08, "sparse_user_item_matrix")
    assert m.shape == (6, 6), (
        f"sparse_user_item_matrix heeft vorm {m.shape}, verwacht (6, 6)"
    )


def test_sparse_user_item_matrix_nnz(week08):
    m = _get(week08, "sparse_user_item_matrix")
    assert m.nnz == 11, (
        f"sparse_user_item_matrix heeft {m.nnz} niet-nul elementen, "
        f"verwacht 11 (één per unieke (user, song)-combinatie in de data)"
    )


def test_als_toy_model_type(week08):
    model = _get(week08, "als_toy_model")
    assert type(model).__name__ == "AlternatingLeastSquares", (
        f"als_toy_model is van type {type(model).__name__}, "
        f"verwacht een implicit AlternatingLeastSquares"
    )
    assert int(model.factors) == 50, (
        f"als_toy_model gebruikt {model.factors} factoren, verwacht 50"
    )


def test_als_toy_model_is_fitted(week08):
    model = _get(week08, "als_toy_model")
    uf = getattr(model, "user_factors", None)
    assert uf is not None, (
        "als_toy_model is niet getraind (user_factors ontbreken); "
        "roep .fit(sparse_user_item_matrix) aan"
    )
    assert uf.shape == (6, 50), (
        f"als_toy_model.user_factors heeft vorm {uf.shape}, "
        f"verwacht (6, 50) (6 users, 50 factoren)"
    )


# ---------------- Meerkeuzevragen ----------------


def test_antwoord_feedback(week08):
    antw = _get(week08, "antwoord_feedback")
    assert str(antw).strip().upper() == "B", (
        f"antwoord_feedback={str(antw)!r} is niet correct"
    )


def test_antwoord_orientatie(week08):
    antw = _get(week08, "antwoord_orientatie")
    assert str(antw).strip().upper() == "B", (
        f"antwoord_orientatie={str(antw)!r} is niet correct"
    )