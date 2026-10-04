"""Loader factory."""
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader


class LoaderFactory:
    """Select a loader based on a file's extension."""
    @staticmethod
    def load(path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents = PyPDFLoader(str(path)).load()
        elif suffix in {".txt", ".text"}:
            documents = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
        elif suffix in {".md", ".markdown"}:
            documents = UnstructuredMarkdownLoader(str(path)).load()
        else:
            raise ValueError(f"Unsupported document type: {path.suffix}")
        for document in documents:
            document.metadata["source"] = path.name
            document.metadata["source_path"] = str(path)
            if "page" in document.metadata:
                document.metadata["page"] = int(document.metadata["page"]) + 1
        return documents
