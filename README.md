# Asistente turístico de Tenerife

Asistente conversacional que responde preguntas sobre una guía turística de Tenerife (`data/TENERIFE.pdf`) **citando
la página de la que sale cada dato**, recuerda la conversación y consulta la predicción del tiempo con una llamada a
función. Usa OpenAI `gpt-4o-mini`, RAG con ChromaDB y un grafo de LangGraph. Trabajo final de *Large Language Models*
(Máster en IA, Cloud Computing y DevOps, Pontia).

## Cómo funciona

```
mensaje ─► router ──┬─► rag      reescribe el seguimiento, busca en Chroma y responde con citas [n]
 (el LLM)           ├─► weather  el LLM llama a get_weather (Open-Meteo, con validación y fallback)
                    └─► other    respuesta fija: no puedo ayudar con eso (no llama al LLM)
```

- **RAG:** cada lugar de la guía es un chunk (los largos se trocean) con zona y páginas como metadatos; las fuentes
  se construyen desde los metadatos, no las escribe el modelo.
- **Router:** el primer nodo es el LLM, que elige `weather`, `rag` u `other`. Ante la duda, `rag`.
- **Memoria:** historial por conversación, recortado a un presupuesto de tokens.
- **`get_weather(fecha)`:** valida la fecha (formato, pasada, más de 15 días), usa Open-Meteo y, si falla, un tiempo
  simulado. Cada intento queda en un log JSON.

## Resultados

20 casos de evaluación repetidos 3 veces con `gpt-4o-mini`: acierto 0,95 · 0,95 · 1,00, con 1–3 s por turno. Solo
falla un seguimiento mal enrutado. Detalle en el [informe](docs/INFORME.md).

## Instalación

Necesitas Python 3.11+, [`uv`](https://docs.astral.sh/uv/), una clave de la API de OpenAI y [Ollama](https://ollama.com)
(los embeddings se calculan en local con `embeddinggemma`).

```bash
cd code
uv venv && uv pip install -e ".[dev,api,ui]"
ollama pull embeddinggemma
cp .env.example .env          # y escribir OPENAI_API_KEY=... en .env
```

`.env` está en `.gitignore`. Solo `config.py` lee variables de entorno; las principales son `LLM_PROVIDER`
(`openai` por defecto), `LLM_MODEL` (`gpt-4o-mini`), `OPENAI_API_KEY`, `TEMPERATURE`, `TOP_P`, `MAX_TOKENS` y
`MAX_HISTORY_TOKENS`.

## Ejecución

Desde `code/`:

```bash
.venv/bin/jupyter lab notebooks/asistente_turistico.ipynb   # notebook principal (ya ejecutado)
.venv/bin/python -m tourist_assistant.ui                    # interfaz web en http://127.0.0.1:7860
.venv/bin/python -m tourist_assistant                       # chat en consola
.venv/bin/python eval/run_e2e.py graph 3                    # repite la evaluación (gasta unos céntimos)
```

La primera ejecución construye el índice en `chroma_db/`; si cambias el chunking, bórralo.

## Estructura

```
src/tourist_assistant/   config, llm, ingest, retrieval, rag, memory, tools, graph, ui, evaluation, __main__
notebooks/               notebook principal
eval/                    casos de evaluación (JSON), script y resultados guardados
docs/                    INFORME.md/.pdf, figures/
data/TENERIFE.pdf        guía turística (corpus)
```

## Limitaciones

- Solo 20 casos de evaluación, escritos por el autor; las comprobaciones comparan texto, no significado.
- Un mensaje sigue una sola ruta: «¿qué tiempo hará mañana y qué veo en el Teide?» se resuelve a medias.
- `get_weather` usa una sola ubicación (Santa Cruz de Tenerife).
- Los embeddings necesitan Ollama en local, así que la interfaz no se despliega tal cual en servicios que no lo ejecutan.
- La interfaz no tiene autenticación ni límite de uso: cada consulta gasta la clave de la API.
- Solo se ha evaluado `gpt-4o-mini`; el proveedor `ollama` para el chat existe pero no se ha vuelto a probar.

Más detalle en el apartado de limitaciones del [informe](docs/INFORME.md).
