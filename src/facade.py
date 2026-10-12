"""Punto único de coordinación para la interfaz Streamlit (patrón Facade)."""

from pathlib import Path

import pandas as pd
from plotly.graph_objects import Figure

from src.agent import LocalDataAgent
from src.data_loader import (
    ANALYSIS_COLUMNS, DATA_PATH, dataset_profile, load_data, load_schema,
)
from src.eda import (
    QUESTIONS, SEGMENTS, AnalysisFactory, agent_context,
    association_contrast, indicator_summary, missing_report,
    prepare_analysis,
)
from src.llm_client import OllamaClient
from src.tools import build_tools
from src.visualizations import build_chart


class SurveyFacade:
    """Oculta a la interfaz la orquestación de carga, análisis y gráficos."""

    def __init__(self, data_path: Path = DATA_PATH):
        self.frame = prepare_analysis(load_data(data_path))

    @property
    def raw(self) -> pd.DataFrame:
        """Columnas originales, sin las variables derivadas."""
        return self.frame[list(ANALYSIS_COLUMNS)]

    def profile(self) -> dict:
        return dataset_profile(self.raw)

    def quality(self) -> pd.DataFrame:
        return missing_report(self.raw)

    def indicators(self) -> pd.DataFrame:
        return indicator_summary(self.frame)

    def schema(self) -> pd.DataFrame:
        return load_schema()

    def analyze(
        self, question: str, kind: str, segment: str | None = None
    ) -> tuple[pd.DataFrame, Figure, dict]:
        """Devuelve tabla, gráfico y contexto basados en el mismo cálculo."""
        result = AnalysisFactory.create(kind).run(
            self.frame, question, segment
        )
        title = QUESTIONS.get(question, question)
        if segment:
            title = f"{title} por {SEGMENTS.get(segment, segment).lower()}"
        chart = build_chart(kind, result, title)
        context = agent_context(
            self.frame, question, result, kind, segment
        )
        return result, chart, context

    def association(self, question: str) -> tuple[pd.DataFrame, Figure]:
        """Cruce entre uso y otra variable, con tabla y barras."""
        result, chart, _ = self.analyze(question, "asociacion")
        return result, chart

    def cohort(
        self, question: str, segment: str
    ) -> tuple[pd.DataFrame, Figure]:
        """Tasa de indicador para cohortes con al menos 100 respuestas."""
        result, chart, _ = self.analyze(question, "cohorte", segment)
        return result, chart

    def contrast(self, question: str) -> dict:
        """Uso diario frente a quienes no usan ni planean usar IA."""
        return association_contrast(self.frame, question)

    def pair_base(self, question: str) -> int:
        """Personas que respondieron AISelect y la pregunta cruzada."""
        return int(self.frame[["AISelect", question]].dropna().shape[0])

    def create_agent(self, client: OllamaClient) -> LocalDataAgent:
        """Construye el agente con las herramientas autorizadas."""
        return LocalDataAgent(client, build_tools(self))
