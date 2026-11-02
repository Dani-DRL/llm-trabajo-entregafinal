"""Memoria de la conversación con control de longitud"""

from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Descarta los turnos más antiguos hasta que el historial cabe en max_tokens (recuento aproximado)"""
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        allow_partial=False,
    )
