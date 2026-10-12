"""Interfaz Streamlit del proyecto final integrador."""

import streamlit as st

from src.data_loader import ANALYSIS_COLUMNS
from src.eda import COMPARISON_QUESTIONS, QUESTIONS, SEGMENTS
from src.facade import SurveyFacade
from src.llm_client import OllamaClient, OllamaError
from src.visualizations import experience_box, experience_histogram

st.set_page_config(page_title="IA en la comunidad de desarrollo",
                   page_icon="🤖", layout="wide")

st.title("IA en la comunidad de desarrollo")
st.caption("Streamlit + Ollama · Encuesta de Desarrolladores Stack Overflow 2025")


@st.cache_resource
def get_facade() -> SurveyFacade:
    """Conserva una sola lectura del CSV durante la sesión."""
    return SurveyFacade()


try:
    facade = get_facade()
except (OSError, ValueError) as error:
    st.error(f"No fue posible cargar el dataset: {error}")
    st.stop()

with st.sidebar:
    st.header("Configuración")
    model = st.text_input("Modelo Ollama", value="llama3.2:3b")
    client = OllamaClient(model=model)
    available, models = client.status()
    ollama_ready = available and model in models
    if ollama_ready:
        st.success("Ollama disponible")
        st.caption("Modelos: " + ", ".join(models[:5]))
    elif available:
        st.warning(f"Ollama responde, pero falta: ollama pull {model}")
    else:
        st.warning("Ollama no detectado")
        st.caption("La exploración funciona sin el agente generativo.")

profile = facade.profile()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Encuestados", f"{profile['rows']:,}")
c2.metric("Variables analizadas", profile["columns"])
c3.metric("Valores nulos", f"{profile['missing']:,}")
c4.metric("Duplicados", profile["duplicates"])

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Datos", "📊 Exploración", "🔗 Comparaciones", "🤖 Agente",
    "🧩 Arquitectura",
])

with tab1:
    st.subheader("Dataset de trabajo")
    st.info(
        "Se analizan 9 de las 170 columnas de la encuesta. Cada porcentaje "
        "usa como base las respuestas válidas de su pregunta."
    )
    st.dataframe(facade.raw.head(25), width="stretch", hide_index=True)
    st.markdown("#### Calidad de datos")
    st.dataframe(facade.quality(), width="stretch", hide_index=True)
    with st.expander("Diccionario de preguntas"):
        schema = facade.schema()
        st.dataframe(schema[schema["qname"].isin(ANALYSIS_COLUMNS)],
                     width="stretch", hide_index=True)

with tab2:
    st.subheader("Exploración")
    st.markdown("#### Indicadores generales")
    st.dataframe(facade.indicators(), width="stretch", hide_index=True)
    question = st.selectbox("Pregunta para distribución", list(QUESTIONS),
                            format_func=QUESTIONS.get)
    table, figure, _ = facade.analyze(question, "distribucion")
    st.plotly_chart(figure, width="stretch")
    st.dataframe(table, width="stretch", hide_index=True)
    st.plotly_chart(experience_histogram(facade.frame), width="stretch")

with tab3:
    st.subheader("Uso de IA frente a otras respuestas")
    relationship = st.selectbox("Cruzar frecuencia de uso con",
                                COMPARISON_QUESTIONS, format_func=QUESTIONS.get)
    association, association_figure = facade.association(relationship)
    st.plotly_chart(association_figure, width="stretch")
    contrast = facade.contrast(relationship)
    st.markdown(f"#### {contrast['resultado']}: uso diario vs. no uso")
    st.dataframe(contrast["grupos"], width="stretch", hide_index=True)
    st.caption(f"Diferencia: {contrast['diferencia_puntos']:+.2f} puntos "
               "porcentuales. Asociación descriptiva, no causal.")

    st.subheader("Uso diario de IA por grupo")
    segment = st.selectbox("Comparar por", list(SEGMENTS),
                           format_func=SEGMENTS.get)
    cohorts, cohort_figure = facade.cohort("AISelect", segment)
    st.plotly_chart(cohort_figure, width="stretch")
    st.dataframe(cohorts, width="stretch", hide_index=True)
    st.plotly_chart(experience_box(facade.frame), width="stretch")

with tab4:
    st.subheader("Agente local para interpretación")
    st.caption("El agente elige una herramienta autorizada, Python calcula "
               "el resultado y Ollama lo interpreta. No recibe el CSV completo.")
    presets = [
        "¿Cuáles son las tasas de uso diario de IA, uso de agentes y "
        "desconfianza en la precisión?",
        "¿Qué porcentaje tiene una actitud favorable hacia la IA entre "
        "quienes la usan a diario y quienes no la usan ni planean usarla? "
        "Indica la base de cada grupo.",
        "¿Cómo varía el uso diario de IA según la experiencia profesional?",
        "¿Qué variables tienen más valores nulos?",
    ]
    choice = st.selectbox("Pregunta sugerida",
                          ["Escribir otra pregunta..."] + presets)
    prompt = st.text_area(
        "Pregunta",
        value="" if choice == "Escribir otra pregunta..." else choice,
        height=100,
    )

    if st.button("Analizar con el agente", type="primary"):
        if not prompt.strip():
            st.warning("Escribe una pregunta.")
        else:
            agent = facade.create_agent(client)
            try:
                with st.spinner("Consultando el modelo local..."):
                    response = agent.ask(
                        prompt.strip(),
                        mode="hibrido" if ollama_ready else "reglas",
                        interpret=ollama_ready,
                    )
            except (OllamaError, ValueError) as error:
                st.error(f"El agente no pudo responder: {error}")
            else:
                if response.answer:
                    st.markdown(response.answer)
                else:
                    st.caption("Ollama no está disponible: se muestra solo "
                               "el resultado calculado.")
                check = response.verification
                details = f"Herramienta: {response.tool} · {response.routing}"
                if check and check.total:
                    details += (f" · cifras verificadas: "
                                f"{len(check.verified)} de {check.total}")
                st.caption(details)
                if check and check.unverified:
                    st.warning("Cifras sin respaldo en el cálculo: "
                               + ", ".join(check.unverified))
                with st.expander("Resultado calculado por Python",
                                 expanded=not response.answer):
                    st.dataframe(response.result.table, width="stretch",
                                 hide_index=True)

with tab5:
    st.subheader("Arquitectura modular")
    st.code(
        """data/data.csv
        ↓
src/data_loader.py
        ↓
src/facade.py  (Facade)
   ↙          ↓            ↘
eda.py      visualizations  agent.py + tools.py
(Factory +    (Plotly)          ↓
 Strategy)                  Ollama local
        ↓
app.py (Streamlit)""",
        language="text",
    )
    st.markdown(
        """
- **Imperativo:** flujo de la interfaz, bucles y validaciones (`association_contrast`, `check_columns`).
- **Funcional:** funciones puras compuestas con `pipe`/`reduce`, `partial`, `map` y `filter` en `eda.py`.
- **Orientado a objetos:** `AnalysisStrategy` y sus subclases, `SurveyFacade`, `LocalDataAgent`, `OllamaClient`.
- **Patrones:** Strategy (análisis intercambiables), Factory (`AnalysisFactory`) y Facade (`SurveyFacade`).
        """
    )

st.markdown("---")
st.caption("Resultados descriptivos de quienes respondieron la encuesta; "
           "no permiten inferir causalidad.")
