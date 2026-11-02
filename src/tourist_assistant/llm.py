"""Creación del modelo de chat según el proveedor configurado"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from tourist_assistant.config import Settings


def get_llm(settings: Settings) -> BaseChatModel:
    """Crea el modelo de chat con temperatura, top_p y máximo de tokens explícitos"""
    params = {"temperature": settings.temperature, "top_p": settings.top_p}
    if settings.llm_provider == "ollama":
        params["num_predict"] = settings.max_tokens
        params["num_ctx"] = settings.num_ctx
        params["base_url"] = settings.ollama_base_url
    else:
        params["max_tokens"] = settings.max_tokens
        params["api_key"] = settings.openai_api_key  # LangChain no lee el .env
    return init_chat_model(settings.llm_model, model_provider=settings.llm_provider, **params)
