"""API model wrappers for closed-weight models (GPT-4o, Claude Sonnet, Gemini).

Tier 2 models in the two-tier experimental design: behavioural metrics
only, no activation access. Used for Experiment 1 (generality claim)
and Phase 3.5 triage.

Tokens are loaded from environment variables or a .env file.
Never store tokens in source code.

Requires: pip install openai anthropic google-genai python-dotenv
"""

from __future__ import annotations

import os
import time
from typing import Optional

from ci_bench.models.base import Model, ModelResponse

# Load .env if present (tokens stored outside source tree).
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass  # dotenv not installed; rely on env vars directly.


def _get_env(key: str) -> str:
    """Retrieve an environment variable or raise a clear error."""
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"Environment variable {key} is not set. "
            f"Create a .env file from .env.example or export it."
        )
    return val


# -----------------------------------------------------------------------
# OpenAI (GPT-4o)
# -----------------------------------------------------------------------

class OpenAIModel(Model):
    """Wrapper for OpenAI chat completions API.

    Usage:
        model = OpenAIModel("gpt-4o-2024-08-06")
        response = model.generate("What is the capital of France?")
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-2024-08-06",
        requests_per_minute: int = 30,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package required. Install with: pip install openai"
            )
        self._model_name = model_name
        self._client = OpenAI(api_key=_get_env("OPENAI_API_KEY"))
        self._min_interval = 60.0 / requests_per_minute
        self._last_call: float = 0.0

    @property
    def model_id(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> ModelResponse:
        # Simple rate limiting.
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        t0 = time.time()

        kwargs: dict = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""

        self._last_call = time.time()

        return ModelResponse(
            text=text,
            model_id=self._model_name,
            prompt_template="",
            prompt_text=prompt,
            temperature=temperature,
            metadata={
                "generation_time_s": round(time.time() - t0, 3),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                },
                "finish_reason": response.choices[0].finish_reason,
            },
        )


# -----------------------------------------------------------------------
# Anthropic (Claude Sonnet)
# -----------------------------------------------------------------------

class AnthropicModel(Model):
    """Wrapper for Anthropic messages API.

    Usage:
        model = AnthropicModel("claude-sonnet-4-20250514")
        response = model.generate("What is the capital of France?")
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        requests_per_minute: int = 30,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package required. Install with: pip install anthropic"
            )
        self._model_name = model_name
        self._client = anthropic.Anthropic(api_key=_get_env("ANTHROPIC_API_KEY"))
        self._min_interval = 60.0 / requests_per_minute
        self._last_call: float = 0.0

    @property
    def model_id(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> ModelResponse:
        # Simple rate limiting.
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        t0 = time.time()

        # Anthropic API: temperature 0 is not allowed; use a very small value.
        api_temp = max(temperature, 0.01)

        response = self._client.messages.create(
            model=self._model_name,
            max_tokens=max_tokens,
            temperature=api_temp,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""

        self._last_call = time.time()

        return ModelResponse(
            text=text,
            model_id=self._model_name,
            prompt_template="",
            prompt_text=prompt,
            temperature=temperature,
            metadata={
                "generation_time_s": round(time.time() - t0, 3),
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
            },
        )


# -----------------------------------------------------------------------
# Google (Gemini)
# -----------------------------------------------------------------------

class GeminiModel(Model):
    """Wrapper for Google Gemini API via the google-genai SDK.

    Usage:
        model = GeminiModel("gemini-2.0-flash")
        response = model.generate("What is the capital of France?")
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        requests_per_minute: int = 30,
    ) -> None:
        try:
            from google import genai
        except ImportError:
            raise RuntimeError(
                "google-genai package required. Install with: "
                "pip install google-genai"
            )
        self._model_name = model_name
        self._client = genai.Client(api_key=_get_env("GEMINI_API_KEY"))
        self._min_interval = 60.0 / requests_per_minute
        self._last_call: float = 0.0

    @property
    def model_id(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> ModelResponse:
        from google.genai import types

        # Simple rate limiting.
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        t0 = time.time()

        # Gemini 2.5 Flash is a thinking model: internal chain-of-thought
        # tokens count against max_output_tokens. With 512 tokens, thinking
        # consumes ~400-500, leaving too few for the visible response.
        # Use 4096 to give both thinking and output room.
        effective_max = max(max_tokens, 4096)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=effective_max,
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )
        text = response.text or ""

        self._last_call = time.time()

        usage_meta = {}
        if response.usage_metadata:
            usage_meta = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "thoughts_tokens": response.usage_metadata.thoughts_token_count,
            }

        return ModelResponse(
            text=text,
            model_id=self._model_name,
            prompt_template="",
            prompt_text=prompt,
            temperature=temperature,
            metadata={
                "generation_time_s": round(time.time() - t0, 3),
                "usage": usage_meta,
            },
        )


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------

def load_api_model(name: str, **kwargs) -> Model:
    """Convenience factory for API models.

    Args:
        name: One of "gpt-4o", "sonnet", "gemini", or a full model identifier.

    Returns:
        An initialised Model instance.
    """
    aliases = {
        "gpt-4o": ("openai", "gpt-4o-2024-08-06"),
        "sonnet": ("anthropic", "claude-sonnet-4-20250514"),
        "gemini": ("google", "gemini-2.5-flash"),
    }

    if name in aliases:
        provider, model_id = aliases[name]
    elif name.startswith("gpt-") or name.startswith("o1") or name.startswith("o3"):
        provider, model_id = "openai", name
    elif name.startswith("claude-"):
        provider, model_id = "anthropic", name
    elif name.startswith("gemini-"):
        provider, model_id = "google", name
    else:
        raise ValueError(
            f"Unknown model: {name!r}. Use 'gpt-4o', 'sonnet', 'gemini', "
            f"or a full model identifier like 'gpt-4o-2024-08-06'."
        )

    if provider == "openai":
        return OpenAIModel(model_id, **kwargs)
    elif provider == "google":
        return GeminiModel(model_id, **kwargs)
    else:
        return AnthropicModel(model_id, **kwargs)
