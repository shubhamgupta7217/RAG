"""Small contracts for components that can be swapped through factories."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> list[Document]: ...


class EmbeddingProvider(ABC):
    @abstractmethod
    def create(self, config: dict[str, Any]) -> Embeddings: ...


class VectorStoreProvider(ABC):
    @abstractmethod
    def create(self, config: dict[str, Any], embeddings: Embeddings) -> VectorStore: ...


class LLMProvider(ABC):
    @abstractmethod
    def create(self, config: dict[str, Any]) -> BaseChatModel: ...
