"""Interfaz web (Gradio): python -m tourist_assistant.ui"""

import time
import uuid
from collections.abc import Iterator

import gradio as gr

from tourist_assistant.config import get_settings
from tourist_assistant.graph import GraphAnswer, build_graph, stream_ask
from tourist_assistant.llm import get_llm
from tourist_assistant.retrieval import get_index

EXAMPLES = [
    "¿Qué playa de Anaga es buena para ver el atardecer?",
    "¿Dónde puedo comer pizza en Garachico?",
    "¿Qué parque acuático merece la pena en el sur?",
    "¿Qué tiempo hará mañana?",
    "¿Cuánto cuesta la entrada al Siam Park?",
]
ROUTE_LABELS = {"rag": "guía turística", "weather": "predicción del tiempo", "other": "ayuda"}


def answer_stream(graph, message: str, thread_id: str) -> Iterator[str]:
    """Emite el texto acumulado de la respuesta y, al final, la respuesta con sus fuentes"""
    text, start = "", time.perf_counter()
    try:
        for part in stream_ask(graph, message, thread_id):
            if isinstance(part, GraphAnswer):
                final = part
            else:
                text += part
                yield text
    except Exception as error:
        warning = (
            f"No he podido completar la respuesta ({type(error).__name__}). "
            "Comprueba la clave de la API y que Ollama esté en marcha, e inténtalo de nuevo."
        )
        yield f"{text}\n\n{warning}" if text else warning
        return
    footer = []
    if final.sources:
        footer.append("**Fuentes:** " + " · ".join(final.sources))
    footer.append(
        f"*Consultado: {ROUTE_LABELS[final.route]} · {time.perf_counter() - start:.1f} s*"
    )
    yield final.text + "\n\n" + "\n\n".join(footer)


def build_app(graph, model_name: str = "") -> gr.ChatInterface:
    """Crea la aplicación de chat; cada pestaña del navegador tiene su propia conversación"""
    threads: dict[str, str] = {}  # sesión del navegador → thread_id

    def respond(message, history, request: gr.Request):
        session = request.session_hash if request else "local"
        if not history:  # chat nuevo o vaciado
            threads[session] = uuid.uuid4().hex
        thread_id = threads.setdefault(session, uuid.uuid4().hex)
        yield from answer_stream(graph, message, thread_id)

    return gr.ChatInterface(
        fn=respond,
        title="Asistente turístico de Tenerife",
        description=(
            "Chat con una guía turística de Tenerife (RAG con citas), predicción del tiempo y "
            f"memoria de la conversación. Modelo: {model_name or 'configurado en .env'}."
        ),
        chatbot=gr.Chatbot(
            placeholder="Pregúntame por playas, rutas, miradores, comida o el tiempo."
        ),
        textbox=gr.Textbox(placeholder="Escribe tu pregunta sobre Tenerife…"),
        examples=EXAMPLES,
        cache_examples=False,
        save_history=False,
    )


def main() -> None:
    settings = get_settings()
    print("Cargando índice y modelo…")
    graph = build_graph(get_index(settings), get_llm(settings), settings.max_history_tokens)
    build_app(graph, settings.llm_model).launch()


if __name__ == "__main__":
    main()
