"""CLI: python chat.py"""
import logging
import sys

from src.config import load_config
from src.pipeline.factory import create_pipeline


def main() -> int:
    try:
        config = load_config()
        logging.basicConfig(level=getattr(logging, str(config.get("app", {}).get("log_level", "INFO")).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        pipeline = create_pipeline(config)
        print("RAG chatbot ready. Enter 'exit' or 'quit' to end.")
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            try:
                answer, sources = pipeline.ask(question)
                print(f"\nAssistant: {answer}")
                if sources:
                    print("\nSources:")
                    for document in sources:
                        source = document.metadata.get("source", "unknown")
                        page = document.metadata.get("page")
                        print(f"- {source}" + (f", page {page}" if page is not None else ""))
                else:
                    print("\nNo matching document chunks were found.")
            except Exception as exc:
                logging.exception("Could not answer question: %s", exc)
                print("Sorry, the request failed. Check the log for details.")
        return 0
    except Exception as exc:
        logging.exception("Chat startup failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
