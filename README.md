# Modular Python RAG Chatbot

A small, reusable retrieval-augmented generation application. Document loading, embeddings, vector storage, retrieval, prompts, and LLM calls are separate modules. Factories select implementations from `config.yaml`, so provider changes do not require edits to the RAG pipeline. The system prompt is maintained separately in `prompts/system.txt` and selected through the `prompt.system_file` setting.

## Architecture

`ingest.py` discovers documents in `data/`, loads PDF/TXT/Markdown pages with source metadata, splits them into chunks, embeds them, and stores them in persistent Chroma. A JSON manifest records each file's SHA-256 hash and vector IDs, so unchanged files are skipped and changed or removed files have their old vectors replaced. `chat.py` uses the same configured embedding and vector store, retrieves the top matching chunks, adds filename and page labels to the prompt, then calls the configured chat model.

The factories implement provider selection (Strategy/Factory pattern); `DocumentLoader`, `EmbeddingProvider`, `VectorStoreProvider`, and `LLMProvider` define simple component contracts. Supported chat providers are Ollama, Hugging Face Inference Endpoints, OpenAI-compatible APIs, and Anthropic. Embeddings support local Hugging Face sentence-transformers or OpenAI embeddings. Chroma is the included persistent vector database.

## Setup

Use Python 3.10 or newer. From this directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
Copy-Item .env.example .env  # PowerShell; on macOS/Linux: cp .env.example .env
```

Put `.pdf`, `.txt`, or `.md` files in the included `data/` folder (subdirectories are scanned too). Then run:

```bash
python ingest.py
python chat.py
```

The first Hugging Face embedding run downloads the selected model. Ollama must be installed and running separately; pull the configured model first, for example `ollama pull llama3.2`.

## Configuration

Edit `config.yaml` to choose model/provider, chunk size and overlap, result count, paths, and provider options. Edit `prompts/system.txt` to change the system instructions. Keep API credentials in `.env`, never in YAML. Only the selected API provider needs a key.

Examples:

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.2
  max_tokens: 1024
embeddings:
  provider: openai
  model: text-embedding-3-small
```

Set `OPENAI_API_KEY` in `.env` for the example above. For Anthropic, select `provider: anthropic`, choose an available Anthropic model, and set `ANTHROPIC_API_KEY`. For hosted Hugging Face, select `provider: huggingface`, set `llm.model` to an inference endpoint model/repository, and set `HUGGINGFACEHUB_API_TOKEN`. Ollama needs no API key; set `providers.ollama.base_url` when its server is not at the default URL. Provider-specific packages are listed in `requirements.txt`.

Use a consistent embedding provider and model for ingestion and chat. If you change the embedding model, remove the existing `vectorstore/` and `vectorstore/ingestion_state.json`, then run ingestion again; vector dimensions and meaning must match. The ingestion state file lets the script skip unchanged files. Supported formats are PDF, plain text, and Markdown.

## Project layout

```text
data/                         Source documents (created automatically)
vectorstore/                  Persistent Chroma DB and ingestion manifest
config.yaml                   Runtime configuration
prompts/system.txt            Editable system prompt
.env.example                  Secret variable template
ingest.py / chat.py           User-facing CLIs
src/loaders/                  PDF, text, and Markdown loader factory
src/embeddings/               Embedding provider factory
src/vectorstores/             Vector database factory
src/llms/                     Chat model provider factory
src/retrieval/                Retriever wrapper
src/prompts/                  Prompt and source formatting
src/pipeline/                 RAG orchestration and assembly
src/ingestion.py              Incremental index management
src/interfaces.py             Swappable component contracts
```
