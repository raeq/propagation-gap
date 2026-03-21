"""Abstract model interface and response dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ModelResponse:
    """A single model response with full provenance.

    Attributes:
        text: The raw text output from the model.
        model_id: Exact model identifier (e.g., "mistralai/Mistral-7B-Instruct-v0.3").
        prompt_template: Name of the prompt template used.
        prompt_text: The fully rendered prompt sent to the model.
        temperature: Sampling temperature.
        timestamp: When the response was generated.
        metadata: Additional fields (tokens used, latency, etc.).
    """

    text: str
    model_id: str
    prompt_template: str
    prompt_text: str
    temperature: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)


class Model(ABC):
    """Abstract base class for all model backends.

    Subclasses implement generate() for single-prompt inference. The
    default generate_batch() calls generate() in a loop; backends with
    native batching should override it.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the exact model identifier string."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ModelResponse:
        """Generate a single response.

        Args:
            prompt: The fully rendered prompt string.
            temperature: Sampling temperature (0.0 = greedy).
            max_tokens: Maximum tokens to generate.

        Returns:
            A ModelResponse with full provenance.
        """
        ...

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> list[ModelResponse]:
        """Generate responses for a batch of prompts.

        Default implementation calls generate() sequentially.
        Override for backends with native batching.

        Args:
            prompts: List of fully rendered prompt strings.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per response.

        Returns:
            List of ModelResponse objects, one per prompt.
        """
        return [
            self.generate(p, temperature=temperature, max_tokens=max_tokens)
            for p in prompts
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id!r})"
