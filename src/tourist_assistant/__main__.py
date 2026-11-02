"""Chat de consola: python -m tourist_assistant"""

from tourist_assistant.config import get_settings
from tourist_assistant.graph import GraphAnswer, build_graph, stream_ask
from tourist_assistant.llm import get_llm
from tourist_assistant.retrieval import get_index


def main() -> None:
    settings = get_settings()
    print("Cargando índice y modelo…")
    graph = build_graph(get_index(settings), get_llm(settings), settings.max_history_tokens)
    print("Asistente turístico de Tenerife. Escribe 'salir' para terminar.")
    print("Ejemplos: ¿Qué parque acuático merece la pena en el sur? · ¿Qué tiempo hará mañana?\n")
    while True:
        message = input("Tú: ").strip()
        if message.lower() in {"salir", "exit", "quit"}:
            break
        if not message:
            continue
        print("\nAsistente: ", end="", flush=True)
        try:
            for part in stream_ask(graph, message):
                if isinstance(part, GraphAnswer):
                    result = part
                else:
                    print(part, end="", flush=True)  # los tokens se ven según se generan
        except Exception as error:
            print(
                f"\nNo he podido responder ({type(error).__name__}). ¿Está el modelo disponible?\n"
            )
            continue
        print()
        if result.sources:
            print("Fuentes: " + " | ".join(result.sources))
        print()


if __name__ == "__main__":
    main()
