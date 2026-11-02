"""Grafo de LangGraph: el LLM decide si el mensaje va al RAG, al tiempo o a una respuesta fija"""

import re
from collections.abc import Iterator
from typing import Annotated, Literal, NamedTuple, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from tourist_assistant.memory import trim_history
from tourist_assistant.rag import answer, standalone_question
from tourist_assistant.tools import weather_system_prompt, weather_tool

ROUTER_PROMPT = """Clasifica el último mensaje del usuario de un asistente turístico de \
Tenerife (ten en cuenta la conversación para los seguimientos):
- weather: pregunta sobre el tiempo o la predicción meteorológica de un día concreto.
- rag: pregunta sobre lugares, playas, rutas, miradores, comida, actividades o consejos \
para visitar Tenerife.
- other: saludos, agradecimientos o cualquier tema que no sea turismo en Tenerife.
Si el mensaje es un seguimiento corto («¿y el sábado?», «¿y el lunes?»), usa la misma ruta \
que el turno anterior del usuario. Si es una pregunta sobre el tiempo, aunque mencione un \
lugar, es weather.
Ante la duda entre rag y other, elige rag: un rechazo educado del RAG cuesta menos que \
ignorar una pregunta turística."""

OTHER_REPLY = """Soy un asistente turístico de Tenerife: puedo ayudarte con lugares, \
playas, rutas, comida y la predicción del tiempo. Fuera de eso no puedo ayudarte. \
Puedes preguntarme, por ejemplo:
- ¿Qué playa de Anaga es buena para ver el atardecer?
- ¿Dónde puedo comer en La Orotava?
- ¿Qué tiempo hará mañana?"""


ANSWER_TAG = "answer"  # marca las llamadas que generan la respuesta final (para el streaming)


class Route(BaseModel):
    destination: Literal["rag", "weather", "other"] = Field(description="Ruta elegida")


class GraphAnswer(NamedTuple):
    text: str
    sources: list[str]
    route: str


LONG_QUESTION_CHARS = 300  # los mensajes más largos se reescriben antes de buscar


def classify(llm, messages: list[AnyMessage], prompt: str = ROUTER_PROMPT) -> str:
    """El LLM decide la ruta del último mensaje; si falla, rag"""
    try:
        return llm.with_structured_output(Route).invoke([("system", prompt), *messages]).destination
    except Exception:
        return "rag"


def _strip_citations(text: str) -> str:
    """Quita los [n] del historial: remiten al contexto de ese turno"""
    return re.sub(r"\s*\[[\d,\s]+\]", "", text)


def build_graph(
    index, llm, max_history_tokens: int = 2000, k: int = 4, tool=weather_tool, checkpointer=None
):
    """Compila el grafo del asistente; el estado se guarda por thread_id"""

    def add_and_trim(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
        return trim_history(add_messages(left, right), max_history_tokens)

    class State(TypedDict, total=False):
        messages: Annotated[list[AnyMessage], add_and_trim]
        route: str
        answer: str
        sources: list[str]

    # solo las llamadas de la respuesta final llevan la etiqueta, para que el streaming no mezcle el router
    answer_llm = llm.with_config(tags=[ANSWER_TAG])
    tool_llm = llm.bind_tools([tool]).with_config(tags=[ANSWER_TAG])

    def router(state: State) -> dict:
        return {"route": classify(llm, state["messages"])}

    def rag(state: State) -> dict:
        *history, question = state["messages"]
        long_question = len(question.content) > LONG_QUESTION_CHARS
        needs_rewrite = bool(history) or long_question
        query = standalone_question(question.content, history, llm) if needs_rewrite else None
        result = answer(
            question.content, index, answer_llm, k=k, history=history, search_query=query
        )
        return {
            "answer": result.text,
            "sources": result.sources,
            "messages": [AIMessage(_strip_citations(result.text))],
        }

    def weather(state: State) -> dict:
        messages = [("system", weather_system_prompt()), *state["messages"]]
        ai = tool_llm.invoke(messages)
        if ai.tool_calls:
            results = [
                ToolMessage(str(tool.invoke(call["args"])), tool_call_id=call["id"])
                for call in ai.tool_calls
            ]
            ai = answer_llm.invoke([*messages, ai, *results])
        return {"answer": ai.content, "sources": [], "messages": [AIMessage(ai.content)]}

    def other(state: State) -> dict:
        return {"answer": OTHER_REPLY, "sources": [], "messages": [AIMessage(OTHER_REPLY)]}

    graph = StateGraph(State)
    graph.add_node("router", router)
    graph.add_node("rag", rag)
    graph.add_node("weather", weather)
    graph.add_node("other", other)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda state: state["route"], ["rag", "weather", "other"])
    for node in ("rag", "weather", "other"):
        graph.add_edge(node, END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def ask(graph, message: str, thread_id: str = "default") -> GraphAnswer:
    """Envía un mensaje a la conversación thread_id"""
    state = graph.invoke(
        {"messages": [HumanMessage(message)]}, {"configurable": {"thread_id": thread_id}}
    )
    return GraphAnswer(state["answer"], state["sources"], state["route"])


def stream_ask(graph, message: str, thread_id: str = "default") -> Iterator[str | GraphAnswer]:
    """Como ask, pero emite el texto a medida que se genera; el último elemento es el GraphAnswer"""
    config = {"configurable": {"thread_id": thread_id}}
    streamed = False
    final_state: dict = {}
    for mode, data in graph.stream(
        {"messages": [HumanMessage(message)]}, config, stream_mode=["messages", "values"]
    ):
        if mode == "messages":
            chunk, metadata = data
            is_answer = ANSWER_TAG in (metadata.get("tags") or [])
            if is_answer and isinstance(chunk.content, str) and chunk.content:
                streamed = True
                yield chunk.content
        else:
            final_state = data
    if not streamed:
        yield final_state["answer"]
    yield GraphAnswer(final_state["answer"], final_state["sources"], final_state["route"])
