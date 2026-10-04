"""Chat model strategy factory."""
from typing import Any
from langchain_core.language_models import BaseChatModel


class LLMFactory:
    @staticmethod
    def create(config: dict[str, Any]) -> BaseChatModel:
        provider = config.get("provider", "ollama").lower()
        model = config.get("model")
        if not model:
            raise ValueError("llm.model must be configured")
        common = {"temperature": float(config.get("temperature", 0.2))}
        max_tokens = config.get("max_tokens")
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            options = config.get("providers", {}).get("ollama", {})
            return ChatOllama(model=model, base_url=options.get("base_url"), **common)
        if provider == "huggingface":
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            options = config.get("providers", {}).get("huggingface", {})
            endpoint = HuggingFaceEndpoint(repo_id=model, task=options.get("task", "text-generation"), temperature=common["temperature"], max_new_tokens=int(max_tokens or 1024))
            return ChatHuggingFace(llm=endpoint)
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            options = config.get("providers", {}).get("openai", {})
            kwargs: dict[str, Any] = {**common, "model": model}
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            if options.get("base_url"):
                kwargs["base_url"] = options["base_url"]
            return ChatOpenAI(**kwargs)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            options = config.get("providers", {}).get("anthropic", {})
            kwargs = {**common, "model": model}
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            if options.get("base_url"):
                kwargs["base_url"] = options["base_url"]
            return ChatAnthropic(**kwargs)
        raise ValueError(f"Unsupported LLM provider: {provider}")
