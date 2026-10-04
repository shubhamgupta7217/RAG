"""Document ingestion with content-hash based incremental updates."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from src.loaders.factory import LoaderFactory

logger = logging.getLogger(__name__)
SUPPORTED = {".pdf", ".txt", ".text", ".md", ".markdown"}


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def ingest(config: dict[str, Any], store: VectorStore) -> tuple[int, int]:
    data_dir: Path = config["paths"]["data_dir"]
    state_file: Path = config["paths"]["state_file"]
    data_dir.mkdir(parents=True, exist_ok=True)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read ingestion state; rebuilding it: %s", exc)
        state = {}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(config.get("chunking", {}).get("size", 1000)),
        chunk_overlap=int(config.get("chunking", {}).get("overlap", 150)),
        add_start_index=True,
    )
    if splitter._chunk_size <= splitter._chunk_overlap:
        raise ValueError("chunking.size must be greater than chunking.overlap")
    added = skipped = 0
    current: set[str] = set()
    for path in sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED):
        relative = path.relative_to(data_dir).as_posix()
        current.add(relative)
        try:
            digest = _digest(path)
            if state.get(relative, {}).get("hash") == digest:
                skipped += 1
                continue
            documents = LoaderFactory.load(path)
            chunks = splitter.split_documents(documents)
            ids = [hashlib.sha256(f"{relative}:{digest}:{i}".encode()).hexdigest() for i in range(len(chunks))]
            previous_ids = state.get(relative, {}).get("ids", [])
            if previous_ids:
                store.delete(ids=previous_ids)
            if chunks:
                store.add_documents(chunks, ids=ids)
            state[relative] = {"hash": digest, "ids": ids}
            added += 1
            logger.info("Indexed %s as %d chunks", relative, len(chunks))
        except Exception:
            logger.exception("Failed to ingest %s", path)
    removed = set(state) - current
    for relative in removed:
        old_ids = state.pop(relative, {}).get("ids", [])
        if old_ids:
            try:
                store.delete(ids=old_ids)
            except Exception:
                logger.exception("Failed to remove stale vectors for %s", relative)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    logger.info("Ingestion complete: %d added/updated, %d unchanged", added, skipped)
    return added, skipped
