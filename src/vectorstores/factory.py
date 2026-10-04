"""Vector store strategy factory."""
from typing import Any
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class VectorStoreFactory:
    @staticmethod
    def create(config: dict[str, Any], embeddings: Embeddings) -> VectorStore:
        provider = config.get("provider", "chroma").lower()
        if provider == "chroma":
            from langchain_chroma import Chroma
            directory = config["paths"]["vectorstore_dir"]
            directory.mkdir(parents=True, exist_ok=True)
            return Chroma(
                collection_name=config.get("vectorstore", {}).get("collection", "rag_documents"),
                embedding_function=embeddings,
                persist_directory=str(directory),
            )
        raise ValueError(f"Unsupported vector store provider: {provider}")
