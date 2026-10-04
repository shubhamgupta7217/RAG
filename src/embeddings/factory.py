"""Embedding strategy factory."""
from typing import Any
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingFactory:
    @staticmethod
    def create(config: dict[str, Any]) -> Embeddings:
        provider = config.get("provider", "huggingface").lower()
        model = config.get("model")
        if provider == "huggingface":
            if not model:
                raise ValueError("embeddings.model must be set for Hugging Face embeddings")
            return HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": config.get("device", "cpu")})
        if provider == "openai":
            if not model:
                raise ValueError("embeddings.model must be set for OpenAI embeddings")
            return OpenAIEmbeddings(model=model)
        raise ValueError(f"Unsupported embedding provider: {provider}")
