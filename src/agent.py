"""Agente local: elige una herramienta, la ejecuta e interpreta con Ollama.

Flujo: pregunta → enrutamiento (Ollama en modo JSON o reglas) →
herramienta autorizada (cálculo en Python) → interpretación con Ollama →
verificación de las cifras citadas contra el contexto calculado.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterator

from src.eda import COMPARISON_QUESTIONS, INDICATOR_FLAGS
from src.llm_client import OllamaClient, OllamaError
from src.tools import Tool, ToolResult


# Palabras clave (sin tildes) para el enrutamiento por reglas.
QUESTION_KEYWORDS = {
    "AISent": ("actitud", "favorable", "opin", "sentimiento"),
    "AIAcc": ("confia", "confianza", "desconf", "precision", "precis"),
    "AIAgents": ("agente",),
}
SEGMENT_KEYWORDS = {
    "WorkExpGroup": ("experiencia", "antiguedad", "seniority"),
    "DevType": ("rol", "roles", "cargo", "puesto", "tipo de desarrollador"),
    "Age": ("edad", "edades", "joven", "jovenes"),
    "MainBranch": ("relacion con la programacion", "estudiante",
                   "aficionado", "hobby"),
}
QUALITY_KEYWORDS = ("nulo", "faltante", "calidad", "vacio", "sin respuesta")
USAGE_KEYWORDS = ("diari", "frecuencia de uso", "quienes usan", "no usan")

NUMBER_PATTERN = re.compile(
    r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)*"
)
TRIVIAL_LIMIT = 10
ROUTING_MODES = {
    "hibrido": "Híbrido: reglas explícitas y Ollama para preguntas abiertas",
    "modelo": "Ollama elige la herramienta (respaldo: reglas)",
    "reglas": "Solo reglas por palabras clave",
}

SYSTEM_PROMPT = (
    "Eres un agente académico de análisis de datos. Responde en español "
    "a la pregunta concreta del usuario. Usa como única fuente el resultado "
    "calculado por la herramienta. No apliques fórmulas, no hagas "
    "operaciones y no inventes ni recalcules cifras. Cita "
    "los HECHOS CLAVE tal como aparecen, con casos y bases. Los "
    "porcentajes por grupo usan como denominador a quienes respondieron "
    "dentro de cada grupo. Si el "
    "contexto no permite responder, explica qué dato falta. Distingue "
    "asociación de causalidad y evita generalizar la encuesta."
)


# ---------------------------------------------------------------------------
# Funciones puras de apoyo
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Minúsculas y sin tildes para comparar palabras clave."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def mentions(text: str, keywords: tuple[str, ...]) -> bool:
    """Indica si alguna palabra clave aparece como inicio de palabra."""
    return any(map(lambda word: re.search(rf"\b{word}", text), keywords))


def detect(text: str, catalog: dict[str, tuple[str, ...]]) -> list[str]:
    """Códigos cuyo grupo de palabras aparece en el texto."""
    return [code for code, words in catalog.items() if mentions(text, words)]


def keyword_route(question: str) -> tuple[str, dict, bool]:
    """Enrutamiento determinista por reglas.

    El tercer valor indica si la pregunta contenía palabras clave
    explícitas; si es ``False`` la regla solo propone un valor por defecto.
    """
    text = normalize(question)
    if mentions(text, QUALITY_KEYWORDS):
        return "calidad", {}, True
    questions = detect(text, QUESTION_KEYWORDS)
    segments = detect(text, SEGMENT_KEYWORDS)
    if len(questions) > 1:
        return "indicadores", {}, True
    target = questions[0] if questions else "AISelect"
    if segments:
        if target in INDICATOR_FLAGS:
            return ("cohorte",
                    {"indicador": target, "segmento": segments[0]}, True)
        return ("comparacion",
                {"pregunta": target, "segmento": segments[0]}, True)
    if target in COMPARISON_QUESTIONS and mentions(text, USAGE_KEYWORDS):
        return "asociacion", {"pregunta": target}, True
    if questions:
        return "distribucion", {"pregunta": target}, True
    return "indicadores", {}, False


def number_candidates(token: str) -> set[float]:
    """Interpretaciones posibles de un número escrito en español o inglés."""
    compact = re.sub(r"[   ]", "", token)
    separators = re.findall(r"[.,]", compact)
    candidates = {float(re.sub(r"[.,]", "", compact))}
    if separators:
        last = max(compact.rfind("."), compact.rfind(","))
        integer = re.sub(r"[.,]", "", compact[:last])
        candidates.add(float(f"{integer}.{compact[last + 1:]}"))
    return candidates


def context_numbers(value) -> Iterator[float]:
    """Recorre recursivamente el contexto y produce todos sus números."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, str):
        yield from map(float, re.findall(r"\d+(?:\.\d+)?", value))
    elif isinstance(value, dict):
        for item in value.values():
            yield from context_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from context_numbers(item)


def close_enough(cited: float, reference: float) -> bool:
    """Admite redondeo a entero o a un decimal."""
    tolerance = 0.5 if cited.is_integer() else 0.051
    return abs(cited - reference) <= tolerance


# ---------------------------------------------------------------------------
# Objetos del agente
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Verification:
    """Cifras de la respuesta encontradas o no en el contexto calculado."""

    verified: tuple[str, ...]
    unverified: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.verified) + len(self.unverified)


@dataclass
class AgentResponse:
    """Traza completa de una consulta al agente."""

    question: str
    tool: str
    arguments: dict
    routing: str
    result: ToolResult
    answer: str | None = None
    verification: Verification | None = None
    notes: list[str] = field(default_factory=list)


def verify_answer(answer: str, context: dict) -> Verification:
    """Comprueba que cada cifra no trivial del texto exista en el contexto."""
    references = list(context_numbers(context))
    tokens = [
        token for token in NUMBER_PATTERN.findall(answer)
        if max(number_candidates(token)) > TRIVIAL_LIMIT
    ]
    found = [
        any(close_enough(value, reference)
            for value in number_candidates(token)
            for reference in references)
        for token in tokens
    ]
    return Verification(
        verified=tuple(t for t, ok in zip(tokens, found) if ok),
        unverified=tuple(t for t, ok in zip(tokens, found) if not ok),
    )


class LocalDataAgent:
    """Coordina modelo local y herramientas autorizadas."""

    def __init__(self, client: OllamaClient, tools: dict[str, Tool],
                 max_corrections: int = 1):
        self.client = client
        self.tools = tools
        self.max_corrections = max_corrections

    def model_route(self, question: str) -> tuple[str, dict]:
        """Pide a Ollama, en modo JSON, una herramienta del catálogo."""
        catalog = [tool.describe() for tool in self.tools.values()]
        messages = [
            {"role": "system", "content": (
                "Eres el enrutador de un agente de análisis de datos. "
                "Elige UNA herramienta del catálogo para responder la "
                "pregunta. Responde solo JSON con la forma "
                '{"herramienta": "<nombre>", "argumentos": {...}}. '
                "Cada argumento es UN código de texto (la clave, no la "
                "etiqueta) de los que aparecen en el catálogo. Ejemplos: "
                '{"herramienta": "cohorte", "argumentos": '
                '{"indicador": "AISelect", "segmento": "Age"}} para uso '
                "diario por edad; "
                '{"herramienta": "asociacion", "argumentos": '
                '{"pregunta": "AIAcc"}} para desconfianza entre quienes '
                "usan IA a diario y quienes no la usan; "
                '{"herramienta": "indicadores", "argumentos": {}} para '
                "tasas generales."
            )},
            {"role": "user", "content": (
                f"CATÁLOGO:\n{json.dumps(catalog, ensure_ascii=False)}"
                f"\n\nPREGUNTA:\n{question}"
            )},
        ]
        choice = json.loads(
            self.client.chat(messages, json_mode=True, max_tokens=120)
        )
        name = choice["herramienta"]
        arguments = choice.get("argumentos") or {}
        return name, self.tools[name].validate(arguments)

    def route(self, question: str,
              mode: str = "hibrido") -> tuple[str, dict, str, list[str]]:
        """Elige herramienta según el modo: hibrido, modelo o reglas.

        En modo híbrido las reglas explícitas tienen prioridad y el modelo
        decide las preguntas abiertas. Si el modelo falla, se usan reglas.
        """
        if mode not in ROUTING_MODES:
            raise ValueError(f"Modo de enrutamiento desconocido: {mode}")
        name, arguments, confident = keyword_route(question)
        notes = []
        if mode == "reglas" or (mode == "hibrido" and confident):
            return name, arguments, "Reglas por palabras clave", notes
        try:
            model_name, model_arguments = self.model_route(question)
            return model_name, model_arguments, "Ollama (JSON)", notes
        except (OllamaError, ValueError, KeyError, TypeError) as error:
            notes.append(
                f"El modelo no eligió una herramienta válida ({error}); "
                "se aplicaron reglas por palabras clave."
            )
        return name, arguments, "Reglas por palabras clave", notes

    def interpretation_messages(self, question: str,
                                response: AgentResponse) -> list[dict]:
        """Arma el mensaje con hechos clave primero y detalle después."""
        context = dict(response.result.context)
        facts = context.pop("hechos_clave", [])
        facts_text = "\n".join(f"- {fact}" for fact in facts) or "- (sin hechos)"
        payload = {
            "herramienta": response.tool,
            "argumentos": response.arguments,
            "detalle": context,
        }
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"HECHOS CLAVE CALCULADOS POR PYTHON:\n{facts_text}"
                "\n\nCONTEXTO ADICIONAL:\n"
                f"{json.dumps(payload, ensure_ascii=False, default=str)}"
                f"\n\nPREGUNTA DEL USUARIO:\n{question}"
            )},
        ]

    def interpret(self, question: str, response: AgentResponse) -> None:
        """Interpreta con Ollama, verifica y corrige una vez si hace falta."""
        messages = self.interpretation_messages(question, response)
        context = response.result.context
        answer = self.client.chat(messages).strip()
        check = verify_answer(answer, context)
        for _ in range(self.max_corrections):
            if not check.unverified:
                break
            messages = messages + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": (
                    "Estas cifras no aparecen en los hechos calculados: "
                    f"{', '.join(check.unverified)}. Reescribe la respuesta "
                    "usando solo cifras de los HECHOS CLAVE, sin calcular "
                    "otras nuevas."
                )},
            ]
            retry = self.client.chat(messages).strip()
            retry_check = verify_answer(retry, context)
            response.notes.append(
                "La primera respuesta citaba cifras sin respaldo "
                f"({', '.join(check.unverified)}); se pidió al modelo "
                "corregirla."
            )
            if len(retry_check.unverified) <= len(check.unverified):
                answer, check = retry, retry_check
        response.answer, response.verification = answer, check

    def ask(self, question: str, mode: str = "hibrido",
            interpret: bool = True) -> AgentResponse:
        """Ejecuta el flujo completo; sin modelo devuelve solo el cálculo."""
        name, arguments, routing, notes = self.route(question, mode)
        response = AgentResponse(
            question=question, tool=name, arguments=arguments,
            routing=routing, result=self.tools[name](**arguments),
            notes=notes,
        )
        if interpret:
            self.interpret(question, response)
        return response
