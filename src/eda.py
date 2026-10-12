"""Funciones analíticas y estrategias intercambiables.

El módulo combina tres paradigmas:

* Funcional: transformaciones puras encadenadas con ``pipe`` (``reduce``),
  funciones de orden superior, ``map``, ``filter`` y ``functools.partial``.
* Imperativo: validaciones y acumulación explícita con bucles ``for`` e
  instrucciones ``if``.
* Orientado a objetos: jerarquía ``AnalysisStrategy`` (Strategy) y
  ``AnalysisFactory`` (Factory).
"""

from abc import ABC, abstractmethod
from functools import partial, reduce
from typing import Callable

import pandas as pd


QUESTIONS = {
    "AISelect": "Uso de herramientas de IA",
    "AIAgents": "Uso de agentes de IA",
    "AISent": "Actitud hacia la IA",
    "AIAcc": "Confianza en las respuestas de IA",
}
SEGMENTS = {
    "Age": "Edad",
    "WorkExpGroup": "Experiencia profesional",
    "DevType": "Rol profesional",
    "MainBranch": "Relación con la programación",
}

EXPERIENCE_LABELS = ("1-4 años", "5-9 años", "10-19 años", "20+ años")
COMPARISON_QUESTIONS = ("AISent", "AIAcc", "AIAgents")
DAILY_USE = "Yes, I use AI tools daily"
NO_USE = "No, and I don't plan to"
DISTRUST = ("Somewhat distrust", "Highly distrust")
MIN_COHORT_BASE = 100

# Indicador binario derivado de cada pregunta (columna creada al preparar).
INDICATOR_FLAGS = {
    "AISelect": "uso_diario",
    "AIAgents": "usa_agentes",
    "AIAcc": "desconfia_ia",
}
INDICATOR_DEFINITIONS = (
    ("Uso diario de IA", "AISelect"),
    ("Uso de agentes", "AIAgents"),
    ("Desconfianza en precisión", "AIAcc"),
)


# ---------------------------------------------------------------------------
# Paradigma funcional: funciones puras y composición
# ---------------------------------------------------------------------------

def pipe(value, *functions: Callable):
    """Aplica funciones en secuencia: pipe(x, f, g) equivale a g(f(x))."""
    return reduce(lambda accumulated, function: function(accumulated),
                  functions, value)


def percentage(numerator: int, denominator: int) -> float:
    """Porcentaje redondeado; devuelve 0 si la base es vacía."""
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def add_experience_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Agrupa los años de experiencia en cohortes ordenadas."""
    return frame.assign(WorkExpGroup=pd.cut(
        frame["WorkExp"], bins=[0, 4, 9, 19, float("inf")],
        labels=EXPERIENCE_LABELS,
    ))


def add_flag(frame: pd.DataFrame, name: str, source: str,
             predicate: Callable[[pd.Series], pd.Series]) -> pd.DataFrame:
    """Función de orden superior: crea un indicador con un predicado."""
    return frame.assign(**{name: predicate(frame[source])})


add_daily_use = partial(
    add_flag, name="uso_diario", source="AISelect",
    predicate=lambda answers: answers.eq(DAILY_USE),
)
add_agent_use = partial(
    add_flag, name="usa_agentes", source="AIAgents",
    predicate=lambda answers: answers.str.startswith(
        "Yes, I use AI agents at work", na=False
    ),
)
add_distrust = partial(
    add_flag, name="desconfia_ia", source="AIAcc",
    predicate=lambda answers: answers.isin(DISTRUST),
)


def prepare_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Compone transformaciones puras sin alterar el DataFrame original."""
    return pipe(
        frame, add_experience_group, add_daily_use,
        add_agent_use, add_distrust,
    )


def indicator_row(frame: pd.DataFrame, definition: tuple[str, str]) -> dict:
    """Calcula un indicador con la base válida de su propia pregunta."""
    label, source = definition
    valid = frame[source].notna()
    cases = int(frame.loc[valid, INDICATOR_FLAGS[source]].sum())
    base = int(valid.sum())
    return {"indicador": label, "casos": cases, "base": base,
            "porcentaje": percentage(cases, base)}


def indicator_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aplica ``indicator_row`` a cada definición mediante ``map``."""
    return pd.DataFrame(
        list(map(partial(indicator_row, frame), INDICATOR_DEFINITIONS))
    )


def missing_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Cuenta respuestas faltantes por columna analizada."""
    return pd.DataFrame({
        "variable": frame.columns,
        "nulos": frame.isna().sum().values,
        "porcentaje": (frame.isna().mean().values * 100).round(2),
    }).sort_values("nulos", ascending=False).reset_index(drop=True)


def large_groups(result: pd.DataFrame,
                 minimum: int = MIN_COHORT_BASE) -> pd.DataFrame:
    """Conserva solo los grupos con base suficiente usando ``filter``."""
    rows = filter(lambda row: row["base"] >= minimum,
                  result.to_dict("records"))
    return pd.DataFrame(list(rows), columns=result.columns)


# ---------------------------------------------------------------------------
# Paradigma imperativo: validación y acumulación explícita
# ---------------------------------------------------------------------------

def association_contrast(frame: pd.DataFrame, question: str) -> dict:
    """Contrasta un resultado definido entre dos frecuencias de uso."""
    outcomes = {
        "AISent": ("Actitud favorable", {"Favorable", "Very favorable"}),
        "AIAcc": ("Desconfianza en precisión", set(DISTRUST)),
        "AIAgents": (
            "Uso de agentes",
            {
                "Yes, I use AI agents at work daily",
                "Yes, I use AI agents at work weekly",
                "Yes, I use AI agents at work monthly or infrequently",
            },
        ),
    }
    if question not in outcomes:
        raise ValueError("No existe un contraste para esta pregunta.")
    label, positives = outcomes[question]
    groups = {"Uso diario": DAILY_USE, "No usa ni planea usar": NO_USE}
    valid = frame[["AISelect", question]].dropna()
    results = []
    for group_label, group_value in groups.items():
        answers = valid.loc[valid["AISelect"].eq(group_value), question]
        cases = int(answers.isin(positives).sum())
        base = int(len(answers))
        results.append({
            "grupo": group_label, "casos": cases, "base": base,
            "porcentaje": percentage(cases, base),
        })
    return {
        "resultado": label,
        "grupos": results,
        "diferencia_puntos": round(
            results[0]["porcentaje"] - results[1]["porcentaje"], 2
        ),
    }


def check_columns(frame: pd.DataFrame, *names: str) -> None:
    """Detiene el análisis si falta alguna columna solicitada."""
    for name in names:
        if name not in frame.columns:
            raise ValueError(f"La columna {name} no existe en los datos.")


# ---------------------------------------------------------------------------
# Paradigma orientado a objetos: patrón Strategy
# ---------------------------------------------------------------------------

class AnalysisStrategy(ABC):
    """Contrato común de los análisis disponibles."""

    @abstractmethod
    def run(self, frame: pd.DataFrame, question: str,
            segment: str | None = None) -> pd.DataFrame:
        """Devuelve una tabla calculada a partir del dataset."""


class DistributionStrategy(AnalysisStrategy):
    """Frecuencia y porcentaje entre quienes respondieron una pregunta."""

    def run(self, frame: pd.DataFrame, question: str,
            segment: str | None = None) -> pd.DataFrame:
        check_columns(frame, question)
        answers = frame[question].dropna()
        counts = answers.value_counts().rename_axis("respuesta")
        result = counts.reset_index(name="respuestas")
        result["porcentaje"] = (
            result["respuestas"] / len(answers) * 100
        ).round(2)
        return result


class ComparisonStrategy(AnalysisStrategy):
    """Distribución completa de una pregunta dentro de cada segmento."""

    def run(self, frame: pd.DataFrame, question: str,
            segment: str | None = None) -> pd.DataFrame:
        if segment is None:
            raise ValueError(
                "La comparación requiere una variable de segmento."
            )
        check_columns(frame, question, segment)
        valid = frame[[segment, question]].dropna()
        counts = (
            valid.groupby([segment, question], observed=True)
            .size().rename("respuestas").reset_index()
        )
        counts.columns = ["segmento", "respuesta", "respuestas"]
        counts["total_segmento"] = counts.groupby("segmento")[
            "respuestas"
        ].transform("sum")
        counts = counts[counts["total_segmento"] >= MIN_COHORT_BASE].copy()
        counts["porcentaje"] = (
            counts["respuestas"] / counts["total_segmento"] * 100
        ).round(2)
        return counts.sort_values(
            ["total_segmento", "segmento", "respuestas"],
            ascending=[False, True, False],
        ).reset_index(drop=True)


class AssociationStrategy(AnalysisStrategy):
    """Cruza frecuencia de uso con otra dimensión de IA."""

    def run(self, frame: pd.DataFrame, question: str,
            segment: str | None = None) -> pd.DataFrame:
        if question not in COMPARISON_QUESTIONS:
            raise ValueError("Seleccione actitud, confianza o agentes.")
        valid = frame[["AISelect", question]].dropna()
        counts = (
            valid.groupby(["AISelect", question], observed=True)
            .size().rename("casos").reset_index()
        )
        counts.columns = ["uso_ia", "respuesta", "casos"]
        counts["base_uso"] = counts.groupby("uso_ia")[
            "casos"
        ].transform("sum")
        counts["porcentaje"] = (
            counts["casos"] / counts["base_uso"] * 100
        ).round(2)
        return counts.sort_values(
            ["uso_ia", "casos"], ascending=[True, False]
        ).reset_index(drop=True)


class CohortStrategy(AnalysisStrategy):
    """Compara un indicador binario entre cohortes con base mínima."""

    def run(self, frame: pd.DataFrame, question: str,
            segment: str | None = None) -> pd.DataFrame:
        if question not in INDICATOR_FLAGS or segment not in SEGMENTS:
            raise ValueError("Indicador o segmento no admitido.")
        flag = INDICATOR_FLAGS[question]
        valid = frame[[question, segment, flag]].dropna()
        grouped = valid.groupby(segment, observed=True).agg(
            casos=(flag, "sum"), base=(flag, "size"),
        ).reset_index().rename(columns={segment: "segmento"})
        grouped = large_groups(grouped)
        grouped["porcentaje"] = [
            percentage(cases, base)
            for cases, base in zip(grouped["casos"], grouped["base"])
        ]
        return grouped.sort_values("base", ascending=False).reset_index(
            drop=True
        )


# ---------------------------------------------------------------------------
# Patrón Factory
# ---------------------------------------------------------------------------

class AnalysisFactory:
    """Construye la estrategia solicitada sin condicionales en la interfaz."""

    _strategies: dict[str, type[AnalysisStrategy]] = {
        "distribucion": DistributionStrategy,
        "comparacion": ComparisonStrategy,
        "asociacion": AssociationStrategy,
        "cohorte": CohortStrategy,
    }

    @classmethod
    def create(cls, kind: str) -> AnalysisStrategy:
        try:
            return cls._strategies[kind]()
        except KeyError as error:
            raise ValueError(f"Análisis desconocido: {kind}") from error

    @classmethod
    def kinds(cls) -> tuple[str, ...]:
        return tuple(cls._strategies)


def agent_context(frame: pd.DataFrame, question: str,
                  result: pd.DataFrame, kind: str,
                  segment: str | None = None) -> dict:
    """Entrega al agente solo métricas verificables y sus denominadores."""
    check_columns(frame, question)
    context = {
        "total_encuestados": int(len(frame)),
        "pregunta": QUESTIONS.get(question, question),
        "variable": question,
        "tipo_analisis": kind,
        "respuestas_validas": int(frame[question].notna().sum()),
        "sin_respuesta": int(frame[question].isna().sum()),
        "nota": "Porcentajes descriptivos; no implican causalidad.",
    }
    if segment:
        context["segmento"] = SEGMENTS.get(segment, segment)
        context["filas_con_ambas_respuestas"] = int(
            frame[[question, segment]].dropna().shape[0]
        )
    context["filas_resultado"] = int(len(result))
    context["filas_enviadas"] = min(30, len(result))
    context["resultados"] = result.head(30).to_dict("records")
    return context
