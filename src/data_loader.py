"""Carga y validación de los archivos de la encuesta."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "data.csv"
SCHEMA_PATH = BASE_DIR / "data" / "schema.csv"
ANALYSIS_COLUMNS = (
    "ResponseId", "MainBranch", "Age", "WorkExp", "DevType",
    "AISelect", "AIAgents", "AISent", "AIAcc",
)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lee únicamente las variables necesarias para el caso analítico."""
    header = pd.read_csv(path, nrows=0).columns
    missing = set(ANALYSIS_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(missing))}")
    frame = pd.read_csv(path, usecols=list(ANALYSIS_COLUMNS), low_memory=False)
    frame["WorkExp"] = pd.to_numeric(frame["WorkExp"], errors="coerce")
    return frame[list(ANALYSIS_COLUMNS)]


def load_schema(path: Path = SCHEMA_PATH) -> pd.DataFrame:
    """Lee el diccionario de preguntas suministrado con los datos."""
    return pd.read_csv(path)


def dataset_profile(frame: pd.DataFrame) -> dict:
    """Calcula indicadores básicos de calidad para las columnas analizadas."""
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing": int(frame.isna().sum().sum()),
        "duplicates": int(frame["ResponseId"].duplicated().sum()),
    }
