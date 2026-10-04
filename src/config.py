"""Configuration loading and validation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when project configuration is invalid."""


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration, resolve paths relative to its directory, and load .env."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    load_dotenv(config_path.parent / ".env")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigurationError("The configuration root must be a mapping.")
    paths = config.setdefault("paths", {})
    for key in ("data_dir", "vectorstore_dir", "state_file"):
        if key in paths:
            value = Path(os.path.expandvars(str(paths[key]))).expanduser()
            paths[key] = value if value.is_absolute() else config_path.parent / value
    prompt = config.setdefault("prompt", {})
    if prompt.get("system_file"):
        prompt_path = Path(os.path.expandvars(str(prompt["system_file"]))).expanduser()
        prompt["system_file"] = prompt_path if prompt_path.is_absolute() else config_path.parent / prompt_path
    return config


def required_env(name: str) -> str:
    """Return a required environment value, with an actionable error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"Missing {name}. Set it in .env or the environment.")
    return value
