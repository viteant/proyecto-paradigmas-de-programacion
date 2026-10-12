# Uso, actitud y confianza hacia la inteligencia artificial en la comunidad de desarrollo

Proyecto final integrador: aplicación modular en Streamlit con agente de IA local y análisis de datos.

## Resumen

Este proyecto presenta una aplicación modular en Streamlit para analizar el uso, la actitud y la confianza hacia la inteligencia artificial (IA) entre las 49 123 personas que respondieron la Encuesta de Desarrolladores 2025 de Stack Overflow. La solución separa la carga de datos, el análisis, las visualizaciones, la interfaz y la integración con IA en módulos independientes. Combina programación funcional, imperativa y orientada a objetos, y aplica los patrones Strategy, Factory y Facade. Un agente local elige una de seis herramientas autorizadas, ejecuta el cálculo en Python e interpreta el resultado con el modelo Llama 3.2 de 3B mediante Ollama. Después verifica que cada cifra citada exista en el cálculo y, si no, solicita una corrección. Los resultados muestran que el 47,09 % de quienes respondieron usa herramientas de IA a diario y que el 87,4 % de ese grupo tiene una actitud favorable, frente al 4,02 % de quienes no las usan ni planean usarlas. La documentación (README y requirements.txt) permite reconstruir el entorno y ejecutar la aplicación localmente.

**Palabras clave:** Streamlit, Ollama, agentes de IA, análisis de datos, paradigmas de programación.

## Introducción

Las herramientas de IA generativa se han incorporado rápidamente al trabajo de desarrollo de software. Sin embargo, su adopción no es homogénea: varía según la experiencia, el rol y la confianza que las personas depositan en sus resultados. La Encuesta de Desarrolladores de Stack Overflow es una de las fuentes públicas más amplias sobre este fenómeno y reúne preguntas específicas sobre frecuencia de uso de IA, actitud, confianza en la precisión y uso de agentes.

El proyecto desarrolla una aplicación en Streamlit que concentra la exploración y el análisis de esas preguntas en una interfaz web. La lógica analítica se mantiene en módulos reutilizables. Se complementa con un agente local que responde preguntas en lenguaje natural, siempre a partir de resultados calculados previamente por Python. El dataset es apropiado porque combina variables categóricas de actitud con variables de segmentación (edad, experiencia, rol) y presenta valores faltantes reales. Esto obliga a tratar con cuidado las bases de cálculo.

## Objetivos

### Objetivo general

Desarrollar una aplicación modular en Streamlit para explorar y analizar el uso de IA en la comunidad de desarrollo, integrando funciones reutilizables y un agente de IA local mediante Ollama para apoyar la interpretación de resultados.

### Objetivos específicos

- Diseñar funciones para la carga, validación, exploración y visualización del dataset.
- Organizar la aplicación en módulos y clases que separen responsabilidades, aplicando los paradigmas funcional, imperativo y orientado a objetos, y los patrones Strategy, Factory y Facade.
- Integrar un modelo local mediante Ollama que elija herramientas autorizadas e interprete resultados calculados, con verificación de las cifras citadas.
- Documentar el entorno, las dependencias y el procedimiento de ejecución para garantizar la reproducibilidad.

## Marco teórico

### Streamlit y análisis de datos

Streamlit permite construir aplicaciones de datos con Python y componentes interactivos, por lo que funciona como capa de presentación de un flujo analítico (Streamlit, s. f.). En este proyecto, la interfaz no contiene cálculos: delega la carga, el análisis y la visualización en módulos especializados. Los cálculos se apoyan en pandas (McKinney, 2022) y los gráficos en Plotly Express (Plotly Technologies, s. f.).

### Ollama y agentes locales

Ollama ejecuta modelos de lenguaje de forma local y los expone mediante una API HTTP (Ollama, s. f.). Se utilizó Llama 3.2 de 3B, un modelo orientado al seguimiento de instrucciones, al resumen y al uso de herramientas (Meta, 2024). El modelo, por sí solo, no constituye un agente: la lógica del agente se implementa en Python y coordina la selección de herramientas, el contexto y la respuesta. Este enfoque sigue la idea de combinar razonamiento y acciones sobre herramientas externas (Yao et al., 2023), restringida aquí a funciones autorizadas.

### Paradigmas de programación

La programación **imperativa** describe paso a paso cómo cambia el estado (bucles, asignaciones, validaciones). La **funcional** compone funciones puras, trata las funciones como valores y evita efectos secundarios; Python la admite con `map`, `filter`, `functools.reduce`, `functools.partial` y expresiones `lambda` (Python Software Foundation, s. f.). La **orientada a objetos** encapsula estado y comportamiento en clases, y aprovecha la herencia y el polimorfismo.

### Patrones de diseño

Strategy define una familia de algoritmos intercambiables bajo un contrato común. Factory delega la creación de objetos a un componente que decide qué clase instanciar. Facade ofrece una interfaz simplificada a un subsistema complejo (Gamma et al., 1994).

## Dataset y problema identificado

**Dataset seleccionado:** Stack Overflow Developer Survey 2025. Fuente: https://survey.stackoverflow.co/2025/ (Stack Overflow, 2025a).

La copia local contiene 49 123 respuestas y 170 columnas en `data.csv`, junto con el diccionario de preguntas en `schema.csv`. La metodología oficial informa 49 009 respuestas en los resultados publicados (Stack Overflow, 2025b), por lo que los porcentajes de este análisis no deben compararse cifra por cifra con el informe oficial. La aplicación carga solo nueve columnas para reducir memoria: el identificador, cuatro variables de segmentación (`MainBranch`, `Age`, `WorkExp`, `DevType`) y cuatro preguntas de IA:

- `AISelect`: frecuencia de uso de herramientas de IA.
- `AIAgents`: uso de agentes de IA en el trabajo.
- `AISent`: actitud hacia la IA.
- `AIAcc`: confianza en la precisión de sus respuestas.

**Problema analítico:** determinar cómo se relaciona la frecuencia de uso de IA con la actitud, la confianza y el uso de agentes, y cómo varía el uso diario entre grupos de edad, experiencia, rol profesional y relación con la programación.

**Justificación técnica:** las variables son mayoritariamente categóricas y ordinales, lo que permite practicar distribuciones, tablas cruzadas y comparaciones entre cohortes. `WorkExp` es numérica y admite histogramas, diagramas de caja y discretización. Las preguntas de IA tienen alrededor de un tercio de respuestas faltantes, de modo que cada cálculo debe declarar su base.

**Tabla 1.** Resumen de calidad del dataset (variables analizadas)

| Variable | Tipo | Nulos | % nulos |
|---|---|---:|---:|
| AIAgents | Categórica | 17 233 | 35,08 |
| AIAcc | Categórica ordinal | 15 859 | 32,28 |
| AISent | Categórica ordinal | 15 689 | 31,94 |
| AISelect | Categórica ordinal | 15 437 | 31,43 |
| WorkExp | Numérica (años) | 6 279 | 12,78 |
| DevType | Categórica | 5 487 | 11,17 |
| Age | Categórica ordinal | 0 | 0,00 |
| MainBranch | Categórica | 0 | 0,00 |
| ResponseId | Identificador | 0 | 0,00 |

*Nota.* 49 123 filas, sin identificadores duplicados. Un valor faltante puede reflejar una pregunta omitida o no mostrada. Elaboración propia a partir de Stack Overflow (2025a).

## Metodología y arquitectura

La aplicación se organiza en capas que separan la interfaz, los datos, el análisis, la visualización y la integración con el modelo local (Figura 1). La interfaz `app.py` solo se comunica con la fachada `SurveyFacade`. La fachada coordina la carga, crea las estrategias de análisis mediante la fábrica, obtiene el gráfico pertinente y construye el agente con sus herramientas.

**Figura 1.** Arquitectura general de la aplicación

![Arquitectura general de la aplicación](../assets/arquitectura.svg)

*Nota.* Elaboración propia.

### Estructura del repositorio

```text
proyecto_final/
├── app.py                 interfaz Streamlit
├── README.md
├── requirements.txt
├── .gitignore
├── assets/                diagrama y capturas
├── data/                  schema.csv (data.csv se descarga aparte)
└── src/
    ├── __init__.py
    ├── data_loader.py     carga, validación y perfil
    ├── eda.py             funciones analíticas, Strategy y Factory
    ├── visualizations.py  gráficos Plotly y selección por tipo de análisis
    ├── facade.py          Facade para la interfaz
    ├── tools.py           herramientas autorizadas del agente
    ├── agent.py           agente: enrutamiento, interpretación y verificación
    └── llm_client.py      cliente HTTP de Ollama
```

`data_loader.py` valida que existan las columnas necesarias antes de leer el CSV. `eda.py` contiene todo el cálculo y no depende de Streamlit. `visualizations.py` solo transforma tablas en figuras. `tools.py` expone los cálculos al agente con argumentos restringidos. `llm_client.py` encapsula el HTTP con Ollama y traduce sus errores a una excepción propia, `OllamaError`. Las tablas de la interfaz, los gráficos y el contexto del agente salen de la misma llamada a la fachada, lo que evita duplicar la lógica de cálculo.

**Tabla 2.** Aplicación de paradigmas y patrones de diseño

| Elemento | Implementación | Ubicación |
|---|---|---|
| Paradigma funcional | `pipe` compone transformaciones puras con `reduce`; `add_flag` es una función de orden superior especializada con `partial`; `map` en `indicator_summary`; `filter` en `large_groups`; diccionario de funciones `CHART_BUILDERS`; generador recursivo `context_numbers` | `eda.py`, `visualizations.py`, `agent.py` |
| Paradigma imperativo | Bucles `for` con acumulación en `association_contrast`; validación en `check_columns` y `Tool.validate`; flujo de control de la interfaz | `eda.py`, `tools.py`, `app.py` |
| Orientado a objetos | Clase abstracta `AnalysisStrategy` con cuatro subclases (herencia y polimorfismo); `SurveyFacade`, `OllamaClient`, `LocalDataAgent`; `dataclasses` `Tool`, `ToolResult`, `Verification`, `AgentResponse` | `eda.py`, `facade.py`, `llm_client.py`, `agent.py`, `tools.py` |
| Strategy | `DistributionStrategy`, `ComparisonStrategy`, `AssociationStrategy` y `CohortStrategy` implementan `run(frame, question, segment)` | `eda.py` |
| Factory | `AnalysisFactory.create(kind)` instancia la estrategia a partir de su nombre | `eda.py` |
| Facade | `SurveyFacade` ofrece `analyze`, `association`, `cohort`, `contrast`, `quality` y `create_agent` | `facade.py` |

*Nota.* Elaboración propia.

El siguiente fragmento muestra la combinación de paradigmas en la preparación de datos: una función de orden superior especializada con `partial` y una composición con `reduce`.

```python
def pipe(value, *functions):
    return reduce(lambda acc, function: function(acc), functions, value)

add_daily_use = partial(add_flag, name="uso_diario", source="AISelect",
                        predicate=lambda answers: answers.eq(DAILY_USE))

def prepare_analysis(frame):
    return pipe(frame, add_experience_group, add_daily_use,
                add_agent_use, add_distrust)
```

## Desarrollo de la aplicación

### Funciones analíticas

- **`prepare_analysis(frame)`.** Recibe el DataFrame original y devuelve una copia con la cohorte de experiencia (1-4, 5-9, 10-19 y 20 o más años) y tres indicadores binarios: uso diario, uso de agentes y desconfianza. No modifica el original.
- **`indicator_summary(frame)`.** Aplica `indicator_row` a cada definición mediante `map`. Devuelve casos, base y porcentaje de cada indicador, con la base propia de cada pregunta.
- **`association_contrast(frame, question)`.** Compara de forma imperativa a quienes usan IA a diario con quienes no la usan ni planean usarla. Para cada grupo informa casos, base, porcentaje y la diferencia en puntos porcentuales.
- **`CohortStrategy.run(frame, question, segment)`.** Calcula la tasa del indicador dentro de cada grupo y, con `filter`, conserva solo los grupos con al menos 100 respuestas, para evitar porcentajes inestables.
- **`missing_report(frame)`.** Cuenta y ordena los valores faltantes por variable (Tabla 1).

### Selección de gráficos

El gráfico responde al tipo de variable y al objetivo:

- Barras horizontales para la distribución de una pregunta categórica.
- Histograma para la distribución de los años de experiencia.
- Barras apiladas para el cruce entre la frecuencia de uso y otra respuesta.
- Barras agrupadas para la distribución completa por segmento (herramienta `comparacion` del agente).
- Barras para las tasas por cohorte.
- Diagrama de caja para la dispersión de la experiencia según la frecuencia de uso.

`build_chart(kind, ...)` elige el gráfico de cada tipo de análisis a partir de un diccionario de funciones, sin condicionales.

### Interfaz Streamlit

La interfaz sigue la estructura del demo de clase. La barra lateral permite configurar el modelo de Ollama e indica si el servicio está disponible. En la parte superior hay cuatro métricas: encuestados, variables, nulos y duplicados. Debajo hay cinco pestañas:

- **Datos:** muestra, calidad y diccionario de preguntas.
- **Exploración:** indicadores generales, distribución de la pregunta elegida e histograma de experiencia.
- **Comparaciones:** cruce de la frecuencia de uso con otra pregunta, contraste entre uso diario y no uso, y uso diario por grupo.
- **Agente:** pregunta sugerida o libre, respuesta y tabla calculada.
- **Arquitectura:** diagrama de módulos, paradigmas y patrones.

Cada tabla muestra el numerador y la base de sus porcentajes.

**Figura 2.** Interfaz principal de la aplicación

[Insertar captura de la pantalla principal con las métricas y la pestaña Exploración.]

*Nota.* La captura muestra las métricas generales y las pestañas de la aplicación.

## Integración del agente de Inteligencia Artificial

La clase `LocalDataAgent` recibe un cliente de Ollama y un registro de herramientas autorizadas. Cada herramienta (`Tool`) envuelve un cálculo de la fachada y declara los valores permitidos de sus argumentos (Tabla 3).

**Tabla 3.** Herramientas autorizadas del agente

| Herramienta | Argumentos permitidos | Cálculo que ejecuta |
|---|---|---|
| `indicadores` | — | Tasas de uso diario, uso de agentes y desconfianza |
| `calidad` | — | Perfil y nulos por variable |
| `distribucion` | `pregunta`: AISelect, AIAgents, AISent, AIAcc | `DistributionStrategy` |
| `asociacion` | `pregunta`: AISent, AIAcc, AIAgents | `AssociationStrategy` y contraste |
| `cohorte` | `indicador`: AISelect, AIAgents, AIAcc; `segmento`: Age, WorkExpGroup, DevType, MainBranch | `CohortStrategy` |
| `comparacion` | `pregunta` y `segmento` | `ComparisonStrategy` |

*Nota.* Elaboración propia.

El flujo de una consulta tiene cuatro pasos:

1. **Enrutamiento.** En el modo híbrido, que es el que usa la interfaz, unas reglas de palabras clave detectan las intenciones explícitas, como "a diario" frente a "no usa", "según edad" o "nulos". Si la pregunta es abierta, Ollama elige la herramienta en modo JSON a partir del catálogo. Los argumentos propuestos por el modelo se validan contra la lista cerrada; un valor no autorizado activa las reglas como respaldo. La clase también admite los modos "modelo" (solo Ollama) y "reglas".
2. **Ejecución.** La herramienta devuelve una tabla para la interfaz y un contexto compacto. El contexto incluye `hechos_clave` redactados por Python, por ejemplo: "Actitud favorable — Uso diario: 87.4 % (13735 de 15715)". El modelo nunca recibe el CSV.
3. **Interpretación.** El modelo responde en español con los hechos clave como única fuente, sin aplicar fórmulas, con temperatura 0 y semilla fija.
4. **Verificación y corrección.** `verify_answer` extrae las cifras no triviales de la respuesta y comprueba que existan en el contexto, con tolerancia de redondeo. Si alguna no tiene respaldo, el agente devuelve al modelo la lista de esas cifras y le pide reescribir sin calcular otras nuevas. La interfaz indica la herramienta usada y cuántas cifras se verificaron, y advierte cuáles quedaron sin respaldo.

Esta validación surgió de las pruebas. Con contextos grandes, el modelo de 3B mezclaba porcentajes de filas distintas o realizaba operaciones propias. Por ejemplo, afirmó que el 100 % de ambos grupos tenía actitud favorable, o aplicó una fórmula inventada que daba 95,5 %; el verificador marcó esas cifras. Tras reducir el contexto, poner los hechos clave al inicio, prohibir las operaciones y añadir la autocorrección, las cinco consultas de prueba terminaron sin cifras sin respaldo. También se observó que el modelo elegía mal la herramienta en cerca de una de cada cuatro preguntas explícitas. Por eso las reglas tienen prioridad cuando la intención es clara, y el modelo se reserva para preguntas abiertas como "dame un panorama general de los datos".

### Modelo local

Se utilizó `llama3.2:3b` (Ollama 0.21.1), descargado con `ollama pull llama3.2:3b`. En equipos con menos recursos puede usarse `llama3.2:1b` escribiendo su nombre en la barra lateral. Si Ollama no está iniciado o el modelo no está instalado, la barra lateral lo advierte. En ese caso, el agente elige la herramienta por reglas y muestra solo el resultado calculado, sin interpretación.

## Resultados

**Tabla 4.** Indicadores generales de IA

| Indicador | Casos | Base | % |
|---|---:|---:|---:|
| Uso diario de herramientas de IA | 15 863 | 33 686 | 47,09 |
| Uso de agentes de IA en el trabajo (cualquier frecuencia) | 9 868 | 31 890 | 30,94 |
| Desconfianza en la precisión (algo o mucha) | 15 204 | 33 264 | 45,71 |

*Nota.* Cada porcentaje usa como base las respuestas válidas de su pregunta. Elaboración propia.

Casi la mitad de quienes respondieron usa IA a diario, pero una proporción similar desconfía de la precisión de sus respuestas. Uso y confianza no van necesariamente juntos. En cuanto a la actitud, el 59,7 % es favorable o muy favorable (36,78 % y 22,93 %) y el 20,4 % es desfavorable o muy desfavorable.

**Tabla 5.** Contraste entre uso diario y no uso de IA

| Resultado | Uso diario | No usa ni planea usar | Diferencia (p. p.) |
|---|---:|---:|---:|
| Actitud favorable | 87,40 % (13 735 de 15 715) | 4,02 % (218 de 5 424) | +83,38 |
| Desconfianza en la precisión | 26,50 % (4 145 de 15 643) | 85,83 % (4 617 de 5 379) | −59,33 |
| Uso de agentes | 46,64 % (6 965 de 14 935) | 2,88 % (151 de 5 252) | +43,76 |

*Nota.* Las bases incluyen solo a quienes respondieron ambas preguntas. Son diferencias descriptivas, no efectos causales. Elaboración propia.

La asociación es marcada: quienes usan IA a diario tienen una actitud mucho más favorable y desconfían menos. Aun así, una de cada cuatro personas que la usa a diario desconfía de su precisión, lo que sugiere un uso con supervisión crítica.

**Tabla 6.** Uso diario de IA por experiencia profesional y por edad

| Experiencia | % (casos/base) | Edad | % (casos/base) |
|---|---:|---|---:|
| 1-4 años | 52,79 (3 335/6 317) | 18-24 | 47,17 (2 608/5 529) |
| 5-9 años | 50,37 (3 300/6 551) | 25-34 | 51,10 (5 588/10 936) |
| 10-19 años | 49,46 (4 702/9 506) | 35-44 | 48,72 (4 607/9 456) |
| 20 o más años | 40,44 (3 525/8 716) | 45-54 | 45,08 (2 128/4 720) |
| | | 55-64 | 33,78 (707/2 093) |
| | | 65 o más | 22,25 (168/755) |

*Nota.* Porcentaje de personas de cada grupo que usa IA a diario, entre quienes respondieron `AISelect`. Elaboración propia.

El uso diario disminuye con la experiencia y con la edad, con una caída más pronunciada a partir de los 55 años y de los 20 años de experiencia. Por rol, las tasas más altas aparecen en desarrollo front-end (59,45 %), móvil (56,09 %) y full-stack (52,95 %). Las más bajas corresponden a sistemas embebidos (29,06 %) y aplicaciones de escritorio o empresariales (32,26 %). Entre quienes se dedican profesionalmente al desarrollo, la tasa es del 50,63 %, frente al 26,19 % de quienes programan como pasatiempo.

**Tabla 7.** Resultados principales del análisis

| Indicador | Resultado | Interpretación |
|---|---|---|
| Registros analizados | 49 123 filas, 9 variables | Sin identificadores duplicados |
| Variable con más nulos | AIAgents (35,08 %) | Las preguntas de IA tienen bases distintas; se informa la base de cada cálculo |
| Relación destacada | Actitud favorable: 87,40 % con uso diario frente a 4,02 % sin uso | Asociación fuerte, descriptiva |
| Consulta al agente | "¿Qué porcentaje tiene una actitud favorable [...] entre quienes la usan a diario y quienes no la usan?" | El agente eligió `asociacion` y respondió 87,4 % (13 735 de 15 715) frente a 4,02 % (218 de 5 424), con 83,38 puntos de diferencia; 9 de 9 cifras verificadas |

*Nota.* Los cálculos son objetivos y reproducibles. La respuesta del agente es una interpretación generada por el modelo y verificada contra esos cálculos. Elaboración propia.

## Reproducibilidad y despliegue

El código está en un repositorio de GitHub (Anexo B). El README documenta la versión de Python, la creación y activación del entorno virtual, la instalación desde `requirements.txt`, la obtención del dataset, la instalación de Ollama, la descarga del modelo y la ejecución:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
python -m streamlit run app.py
```

El archivo `data.csv` (134 MB) no se versiona porque supera el límite de GitHub. El README explica cómo descargarlo y renombrarlo, mientras que `schema.csv` sí está incluido. El repositorio tampoco incluye el modelo de Ollama. Las dependencias tienen versiones acotadas, y el modelo se consulta con temperatura 0 y semilla fija para que las respuestas sean repetibles. La ejecución es local; no se realizó despliegue en la nube, porque el agente requiere el servicio local de Ollama.

## Conclusiones

- La separación en módulos y el patrón Facade permitieron que la interfaz se limitara a presentar resultados. Las tablas, los gráficos y el contexto del agente provienen del mismo cálculo, lo que evita inconsistencias.
- Los tres paradigmas cumplen funciones complementarias: el funcional, para transformaciones puras y componibles; el imperativo, para validaciones y acumulaciones explícitas; y el orientado a objetos, para encapsular las estrategias, el cliente y el agente. Strategy y Factory facilitaron agregar análisis sin modificar la interfaz ni el agente.
- Un modelo local pequeño es útil para redactar interpretaciones, pero no es confiable para elegir herramientas ni para citar cifras sin controles. Las herramientas con argumentos restringidos, los hechos redactados por Python y la verificación de cifras mantienen al agente como apoyo del análisis y no como sustituto.
- En la encuesta, el uso diario de IA alcanza al 47,09 % de quienes respondieron y se asocia con una actitud más favorable y menor desconfianza. Es menor entre las personas de más edad y experiencia. Los resultados son descriptivos y no representan necesariamente a toda la población de desarrolladores.
- Como trabajo futuro se propone ponderar las respuestas, aplicar pruebas de independencia (chi cuadrado) a los cruces y evaluar modelos locales de mayor tamaño para el enrutamiento.

## Referencias

Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design patterns: Elements of reusable object-oriented software*. Addison-Wesley.

McKinney, W. (2022). *Python for data analysis: Data wrangling with pandas, NumPy, and Jupyter* (3.ª ed.). O'Reilly Media.

Meta. (2024). *Llama 3.2 model card*. https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md

Ollama. (s. f.). *llama3.2*. Recuperado el [dd] de [mes] de 2026, de https://ollama.com/library/llama3.2

Plotly Technologies. (s. f.). *Plotly Express in Python*. Recuperado el [dd] de [mes] de 2026, de https://plotly.com/python/plotly-express/

Python Software Foundation. (s. f.). *Functional programming HOWTO*. Recuperado el [dd] de [mes] de 2026, de https://docs.python.org/3/howto/functional.html

Stack Overflow. (2025a). *2025 Stack Overflow Developer Survey* [Conjunto de datos]. https://survey.stackoverflow.co/2025/

Stack Overflow. (2025b). *Methodology: 2025 Stack Overflow Developer Survey*. https://survey.stackoverflow.co/2025/methodology

Streamlit. (s. f.). *Streamlit documentation*. Recuperado el [dd] de [mes] de 2026, de https://docs.streamlit.io/

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. En *International Conference on Learning Representations*. https://arxiv.org/abs/2210.03629

## Anexos

### Anexo A. Capturas de funcionamiento

Incluir solo capturas legibles y recortadas, guardadas en `assets/`:

- **Figura A1.** Barra lateral con Ollama y `llama3.2:3b` disponibles.
- **Figura A2.** Pestaña Datos: métricas, muestra, calidad y diccionario de variables.
- **Figura A3.** Pestaña Exploración: indicadores y distribución de una pregunta.
- **Figura A4.** Pestaña Comparaciones: cruce de uso con actitud y contraste.
- **Figura A5.** Pestaña Comparaciones: uso diario por grupo de edad o experiencia.
- **Figura A6.** Pestaña Agente: respuesta de Ollama, herramienta usada, verificación de cifras y tabla calculada.

### Anexo B. Repositorio de GitHub

Enlace: [https://github.com/USUARIO/REPOSITORIO]

Verificar que el repositorio sea público o que se abra sin solicitar permisos adicionales.

### Anexo C. Video demostrativo

Enlace: [https://youtu.be/ID_DEL_VIDEO] (duración máxima: 5 minutos)

Guion sugerido para el video:

1. Inicio con `streamlit run app.py` y Ollama activo (0:00-0:30).
2. Carga del dataset y pestaña Datos (0:30-1:00).
3. Exploración y Comparaciones, cambiando la pregunta y el segmento (1:00-2:45).
4. Agente: una consulta sugerida, la respuesta, la herramienta usada, la verificación y la tabla calculada (2:45-4:15).
5. Arquitectura, paradigmas y patrones (4:15-5:00).
