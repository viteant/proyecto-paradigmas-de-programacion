"""Herramientas autorizadas que el agente puede ejecutar.

Cada herramienta envuelve un cálculo de Python y declara los valores
permitidos de sus argumentos. El modelo solo elige herramienta y
argumentos; nunca calcula cifras por su cuenta.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import pandas as pd

from src.eda import (
    COMPARISON_QUESTIONS, INDICATOR_FLAGS, MIN_COHORT_BASE, QUESTIONS,
    SEGMENTS,
)

if TYPE_CHECKING:
    from src.facade import SurveyFacade


LIMIT_NOTE = (
    "Resultados descriptivos de quienes respondieron la encuesta; "
    "no permiten inferir causalidad ni representar a todos los "
    "desarrolladores."
)


@dataclass(frozen=True)
class ToolResult:
    """Tabla para la interfaz y contexto compacto para el modelo."""

    table: pd.DataFrame
    context: dict


@dataclass(frozen=True)
class Tool:
    """Función autorizada con argumentos restringidos a valores conocidos."""

    name: str
    description: str
    function: Callable[..., ToolResult]
    parameters: dict[str, dict[str, str]] = field(default_factory=dict)

    def validate(self, arguments: dict) -> dict:
        """Acepta código o etiqueta de cada valor; rechaza lo demás."""
        clean = {}
        for name, allowed in self.parameters.items():
            value = arguments.get(name, next(iter(allowed)))
            if isinstance(value, dict):
                value = next(iter(value), "")
            elif isinstance(value, list):
                value = value[0] if value else ""
            value = str(value)
            by_label = {
                label.lower(): code for code, label in allowed.items()
            }
            if value in allowed:
                clean[name] = value
            elif value.lower() in by_label:
                clean[name] = by_label[value.lower()]
            else:
                raise ValueError(
                    f"Valor no autorizado para {self.name}.{name}: {value}"
                )
        return clean

    def __call__(self, **arguments) -> ToolResult:
        return self.function(**self.validate(arguments))

    def describe(self) -> dict:
        """Ficha que se envía al modelo para que elija la herramienta."""
        return {
            "nombre": self.name,
            "descripcion": self.description,
            "argumentos": self.parameters,
        }


def _subset(mapping: dict[str, str], keys) -> dict[str, str]:
    return {key: mapping[key] for key in keys}


def fact_lines(rows: list[dict], label: str, cases: str,
               base: str) -> list[str]:
    """Redacta cada cifra con su numerador y denominador correctos."""
    return [
        f"{row[label]}: {row['porcentaje']} % ({row[cases]} de {row[base]})"
        for row in rows
    ]


def build_tools(facade: "SurveyFacade") -> dict[str, Tool]:
    """Crea el registro de herramientas sobre los cálculos de la fachada."""

    def indicators() -> ToolResult:
        table = facade.indicators()
        return ToolResult(table, {
            "total_encuestados": facade.profile()["rows"],
            "hechos_clave": fact_lines(
                table.to_dict("records"), "indicador", "casos", "base"
            ),
            "indicadores": table.to_dict("records"),
            "definicion": (
                "Cada porcentaje usa como base las respuestas válidas "
                "de su propia pregunta."
            ),
            "limite": LIMIT_NOTE,
        })

    def quality() -> ToolResult:
        table = facade.quality()
        return ToolResult(table, {
            "perfil": facade.profile(),
            "hechos_clave": [
                f"{row['variable']}: {row['nulos']} nulos "
                f"({row['porcentaje']} % de {facade.profile()['rows']})"
                for row in table.to_dict("records")
            ],
            "nulos_por_variable": table.to_dict("records"),
            "nota": (
                "Un valor faltante puede reflejar una pregunta omitida o "
                "no mostrada al encuestado."
            ),
        })

    def distribution(pregunta: str) -> ToolResult:
        table, _, context = facade.analyze(pregunta, "distribucion")
        rows = table.assign(base=context["respuestas_validas"])
        return ToolResult(table, {
            "hechos_clave": fact_lines(
                rows.to_dict("records"), "respuesta", "respuestas", "base"
            ),
            **context,
            "limite": LIMIT_NOTE,
        })

    def association(pregunta: str) -> ToolResult:
        table, _, context = facade.analyze(pregunta, "asociacion")
        contrast = facade.contrast(pregunta)
        return ToolResult(table, {
            "hechos_clave": [
                f"{contrast['resultado']} — {line}"
                for line in fact_lines(
                    contrast["grupos"], "grupo", "casos", "base"
                )
            ] + [
                "Diferencia: "
                f"{contrast['diferencia_puntos']} puntos porcentuales"
            ],
            "grupos_comparados": (
                "Uso diario = 'Yes, I use AI tools daily'; no usa ni "
                "planea usar = 'No, and I don't plan to'."
            ),
            "denominador": (
                "Personas de cada grupo que respondieron ambas preguntas."
            ),
            "pregunta": context["pregunta"],
            "base_con_ambas_respuestas": facade.pair_base(pregunta),
            "nota": "El detalle por respuesta se muestra en la interfaz.",
            "limite": LIMIT_NOTE,
        })

    def cohort(indicador: str, segmento: str) -> ToolResult:
        table, _, context = facade.analyze(indicador, "cohorte", segmento)
        return ToolResult(table, {
            "hechos_clave": fact_lines(
                table.to_dict("records"), "segmento", "casos", "base"
            ),
            **context,
            "indicador_binario": INDICATOR_FLAGS[indicador],
            "base_minima": MIN_COHORT_BASE,
            "definicion_porcentaje": (
                "Entre las personas de cada grupo que respondieron la "
                "pregunta, porcentaje que cumple el indicador. No es la "
                "distribución de grupos entre quienes lo cumplen."
            ),
            "limite": LIMIT_NOTE,
        })

    def comparison(pregunta: str, segmento: str) -> ToolResult:
        table, _, context = facade.analyze(pregunta, "comparacion", segmento)
        return ToolResult(table, {
            **context,
            "base_minima": MIN_COHORT_BASE,
            "denominador": "total_segmento: respuestas válidas del grupo",
            "limite": LIMIT_NOTE,
        })

    tools = (
        Tool("indicadores",
             "Tasas generales (toda la muestra) de uso diario de IA, uso "
             "de agentes y desconfianza en la precisión. No compara "
             "grupos.",
             indicators),
        Tool("calidad",
             "Perfil del dataset y valores nulos por variable.",
             quality),
        Tool("distribucion",
             "Distribución de respuestas de una pregunta de IA.",
             distribution,
             {"pregunta": QUESTIONS}),
        Tool("asociacion",
             "Cruza la frecuencia de uso de IA con actitud, confianza o "
             "uso de agentes. Úsala cuando la pregunta compare a quienes "
             "usan IA a diario con quienes no la usan.",
             association,
             {"pregunta": _subset(QUESTIONS, COMPARISON_QUESTIONS)}),
        Tool("cohorte",
             "Porcentaje que cumple un indicador (uso diario, agentes o "
             "desconfianza) dentro de cada grupo de un segmento.",
             cohort,
             {"indicador": _subset(QUESTIONS, INDICATOR_FLAGS),
              "segmento": SEGMENTS}),
        Tool("comparacion",
             "Distribución completa de una pregunta dentro de cada grupo "
             "de un segmento.",
             comparison,
             {"pregunta": QUESTIONS, "segmento": SEGMENTS}),
    )
    return {tool.name: tool for tool in tools}
