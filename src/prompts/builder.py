"""Prompt template and source-aware context formatting."""
from typing import Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


def build_prompt(config: dict[str, Any]) -> ChatPromptTemplate:
    prompt_config = config.get("prompt", {})
    prompt_file = prompt_config.get("system_file")
    if prompt_file:
        try:
            system = prompt_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"Could not read system prompt file {prompt_file}: {exc}") from exc
    else:
        system = "Answer using the provided context."
    return ChatPromptTemplate.from_messages([
        ("system", system + "\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])


def format_context(documents: list[Document]) -> str:
    parts: list[str] = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")
        label = f"{source}, page {page}" if page is not None else str(source)
        parts.append(f"[Source: {label}]\n{document.page_content}")
    return "\n\n---\n\n".join(parts)
