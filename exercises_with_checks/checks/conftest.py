"""Shared pytest fixture: voert het student-notebook uit en geeft de namespace terug."""
import os
import nbformat
import pytest
from nbclient import NotebookClient
from pathlib import Path

_NOTEBOOKS = {
    "week01": ("week01", "knn_checkpoint_opgave.ipynb", "knn_checkpoint_oplossing.ipynb"),
    "week02": ("week02", "svm_checkpoint_opgave.ipynb", "svm_checkpoint_oplossing.ipynb"),
    "week03": ("week03", "decision_trees_oefening_checkpoint_opgave.ipynb",
               "decision_trees_oefening_checkpoint_oplossing.ipynb"),
    "week04": ("week04", "arima_checkpoint_opgave.ipynb", "arima_checkpoint_oplossing.ipynb"),
    "week05": ("week05", "herhaling_classification_checkpoint_opgave.ipynb",
               "herhaling_classification_checkpoint_oplossing.ipynb"),
    "week06": ("week06", "kmeans_dbscan_checkpoint_opgave.ipynb",
               "kmeans_dbscan_checkpoint_oplossing.ipynb"),
    "week07": ("week07", "dim_reductie_checkpoint_opgave.ipynb",
               "dim_reductie_checkpoint_oplossing.ipynb"),
}


def _notebook_path(week: str) -> Path:
    """Pad naar het te testen notebook.

    Standaard wordt het opgave-notebook gebruikt (zoals de student het indient).
    Met de env-var CHECKPOINT_SOLUTIONS=1 wordt de modeloplossing gebruikt,
    bijvoorbeeld voor de eindvalidatie (stap 8 van de skill).
    """
    folder, opgave, oplossing = _NOTEBOOKS[week]
    base = Path(__file__).resolve().parents[1] / folder
    if os.environ.get("CHECKPOINT_SOLUTIONS", "0") == "1":
        return base / "solutions" / oplossing
    return base / opgave


def _notebook_namespace(week: str) -> dict:
    path = _notebook_path(week)
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3", allow_errors=True,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    ns = {}
    old_cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            try:
                exec(compile(cell.source, f"<{week}:cell>", "exec"), ns)
            except Exception:
                pass  # student-code fouten: variabelen ontbreken dan gewoon -> test faalt
    finally:
        os.chdir(old_cwd)
    return ns


@pytest.fixture(scope="session")
def week01():
    return _notebook_namespace("week01")


@pytest.fixture(scope="session")
def week02():
    return _notebook_namespace("week02")


@pytest.fixture(scope="session")
def week03():
    return _notebook_namespace("week03")


@pytest.fixture(scope="session")
def week04():
    return _notebook_namespace("week04")


@pytest.fixture(scope="session")
def week05():
    return _notebook_namespace("week05")


@pytest.fixture(scope="session")
def week06():
    return _notebook_namespace("week06")


@pytest.fixture(scope="session")
def week07():
    return _notebook_namespace("week07")
