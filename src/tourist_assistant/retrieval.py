"""Embeddings e índice vectorial (Chroma, distancia coseno)"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from tourist_assistant.config import Settings
from tourist_assistant.ingest import chunk_by_place, load_pdf_lines


class TaskPrefixEmbeddings(OllamaEmbeddings):
    """embeddinggemma necesita prefijos distintos para consulta y documento"""

    def embed_query(self, text: str) -> list[float]:
        # se llama a la clase base: su embed_query usaría embed_documents y añadiría el prefijo de documento
        return OllamaEmbeddings.embed_documents(self, [f"task: search result | query: {text}"])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return OllamaEmbeddings.embed_documents(self, [f"title: none | text: {t}" for t in texts])


def get_embeddings(settings: Settings, task_prefixes: bool = True) -> OllamaEmbeddings:
    """Embeddings de Ollama, con prefijos de tarea"""
    cls = TaskPrefixEmbeddings if task_prefixes else OllamaEmbeddings
    return cls(model=settings.embed_model, base_url=settings.ollama_base_url)


def build_index(
    docs: list[Document], embeddings, name: str = "tenerife", persist_dir: Path | None = None
) -> Chroma:
    """Indexa los documentos en una colección Chroma"""
    return Chroma.from_documents(
        docs,
        embeddings,
        collection_name=name,
        persist_directory=str(persist_dir) if persist_dir else None,
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_index(settings: Settings, name: str = "tenerife") -> Chroma:
    """Abre el índice persistente; si está vacío, lo construye con la guía"""
    index = Chroma(
        collection_name=name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    if not index.get()["ids"]:
        index.add_documents(chunk_by_place(load_pdf_lines(settings.pdf_path)))
    return index
