"""Retrieval-augmented generation orchestration."""
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from src.prompts.builder import build_prompt, format_context
from src.retrieval.retriever import Retriever


class RAGPipeline:
    def __init__(self, store: VectorStore, llm: BaseChatModel, config: dict[str, Any]) -> None:
        self.retriever = Retriever(store, int(config.get("retrieval", {}).get("top_k", 4)))
        self.chain = build_prompt(config) | llm | StrOutputParser()

    def ask(self, question: str) -> tuple[str, list[Any]]:
        documents = self.retriever.retrieve(question)
        answer = self.chain.invoke({"context": format_context(documents), "question": question})
        return answer, documents
