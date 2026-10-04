"""Assemble a configured RAG pipeline."""
from typing import Any
from src.embeddings.factory import EmbeddingFactory
from src.llms.factory import LLMFactory
from src.vectorstores.factory import VectorStoreFactory
from src.pipeline.rag import RAGPipeline


def create_pipeline(config: dict[str, Any]) -> RAGPipeline:
    embeddings = EmbeddingFactory.create(config.get("embeddings", {}))
    store = VectorStoreFactory.create(config, embeddings)
    llm_config = dict(config.get("llm", {}))
    llm_config["providers"] = config.get("providers", {})
    llm = LLMFactory.create(llm_config)
    return RAGPipeline(store, llm, config)
