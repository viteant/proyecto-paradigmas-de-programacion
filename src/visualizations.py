"""Gráficos Plotly adaptados a respuestas categóricas."""

from typing import Callable

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure



def distribution_chart(result: pd.DataFrame, title: str):
    """Presenta porcentajes y recuentos de una pregunta."""
    chart = px.bar(
        result, x="porcentaje", y="respuesta", orientation="h",
        hover_data=["respuestas"], title=title,
        labels={"porcentaje": "% de respuestas válidas", "respuesta": ""},
    )
    chart.update_layout(yaxis={"categoryorder": "total ascending"})
    return chart


def comparison_chart(result: pd.DataFrame, title: str):
    """Compara respuestas porcentuales dentro de cada segmento."""
    chart = px.bar(
        result, x="segmento", y="porcentaje", color="respuesta",
        barmode="group", hover_data=["respuestas", "total_segmento"],
        title=title,
        labels={"segmento": "Grupo", "porcentaje": "% dentro del grupo"},
    )
    chart.update_layout(xaxis_tickangle=-35)
    return chart


def association_chart(result: pd.DataFrame, title: str):
    """Muestra la mezcla de respuestas dentro de cada frecuencia de uso."""
    chart = px.bar(
        result, x="uso_ia", y="porcentaje", color="respuesta",
        barmode="stack", hover_data=["casos", "base_uso"],
        title=title,
        labels={"uso_ia": "Frecuencia de uso de IA",
                "porcentaje": "% dentro de la frecuencia"},
    )
    chart.update_layout(xaxis_tickangle=-25)
    return chart


def cohort_chart(result: pd.DataFrame, title: str):
    """Presenta tasas por cohorte con el tamaño de la base visible."""
    visible = result.head(10)
    return px.bar(
        visible, x="segmento", y="porcentaje",
        hover_data=["casos", "base"],
        text="porcentaje", title=title,
        labels={"segmento": "Grupo", "porcentaje": "% del grupo"},
    )




def experience_histogram(frame: pd.DataFrame):
    """Muestra la forma de la distribución de años profesionales."""
    valid = frame[frame["WorkExp"].between(1, 60)]
    return px.histogram(
        valid, x="WorkExp", nbins=30, title="Años de experiencia",
        labels={"WorkExp": "Años", "count": "Personas"},
    )


def experience_box(frame: pd.DataFrame):
    """Compara mediana y dispersión de experiencia por uso de IA."""
    valid = frame[
        frame["WorkExp"].between(1, 60) & frame["AISelect"].notna()
    ]
    return px.box(
        valid, x="AISelect", y="WorkExp", points=False,
        title="Experiencia según frecuencia de uso de IA",
        labels={"AISelect": "Frecuencia de uso", "WorkExp": "Años"},
    )







# Cada tipo de análisis de AnalysisFactory tiene su gráfico. El diccionario
# de funciones reemplaza una cadena de condicionales (despacho funcional).
CHART_BUILDERS: dict[str, Callable[[pd.DataFrame, str], Figure]] = {
    "distribucion": distribution_chart,
    "comparacion": comparison_chart,
    "asociacion": association_chart,
    "cohorte": cohort_chart,
}


def build_chart(kind: str, result: pd.DataFrame, title: str) -> Figure:
    """Devuelve el gráfico pertinente para el tipo de análisis."""
    try:
        return CHART_BUILDERS[kind](result, title)
    except KeyError as error:
        raise ValueError(f"No hay gráfico para: {kind}") from error
