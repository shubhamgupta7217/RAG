"""CLI: python ingest.py"""
import logging
import sys

from src.config import load_config
from src.embeddings.factory import EmbeddingFactory
from src.vectorstores.factory import VectorStoreFactory
from src.ingestion import ingest


def main() -> int:
    try:
        config = load_config()
        logging.basicConfig(level=getattr(logging, str(config.get("app", {}).get("log_level", "INFO")).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        config["paths"]["data_dir"].mkdir(parents=True, exist_ok=True)
        embeddings = EmbeddingFactory.create(config.get("embeddings", {}))
        store = VectorStoreFactory.create(config, embeddings)
        added, skipped = ingest(config, store)
        print(f"Ingestion complete: {added} file(s) added/updated, {skipped} unchanged.")
        return 0
    except Exception as exc:
        logging.exception("Ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
