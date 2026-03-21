"""Configuration loader for CI-Bench experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config as a dict.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load a model config and validate required fields.

    Args:
        path: Path to the model YAML file.

    Returns:
        Parsed model config.

    Raises:
        ValueError: If required fields are missing.
    """
    config = load_config(path)
    required = {"model_id", "backend"}
    missing = required - set(config.keys())
    if missing:
        raise ValueError(
            f"Model config {path} missing required fields: {missing}"
        )
    return config


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config and resolve model config paths.

    Args:
        path: Path to the experiment YAML file.

    Returns:
        Parsed experiment config with model configs loaded inline.
    """
    config = load_config(path)

    # Resolve model config paths relative to the experiment config's parent.
    base_dir = Path(path).parent
    if "models" in config:
        resolved_models = []
        for model_path in config["models"]:
            # Try relative to config file first, then absolute.
            candidate = base_dir / model_path
            if not candidate.exists():
                candidate = Path(model_path)
            resolved_models.append(load_model_config(candidate))
        config["_resolved_models"] = resolved_models

    return config
