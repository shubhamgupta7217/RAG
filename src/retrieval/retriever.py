"""Retriever wrapper."""
from typing import Any
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class Retriever:
    def __init__(self, store: VectorStore, top_k: int = 4) -> None:
        if top_k < 1:
            raise ValueError("retrieval.top_k must be at least 1")
        self._retriever = store.as_retriever(search_kwargs={"k": top_k})

    def retrieve(self, question: str) -> list[Document]:
        return self._retriever.invoke(question)
