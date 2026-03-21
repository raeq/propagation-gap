"""MLX model wrapper for inference on Apple Silicon.

Requires: pip install mlx mlx-lm

This module wraps mlx-lm's load/generate API and implements the
Model base class for CI-Bench evaluation. It also provides the
activation extraction hook used by ci_bench.probes.extract.

NOTE: This code is written for mlx-lm ~0.30.x. The activation
extraction intercepts the inner model's forward pass by temporarily
replacing model.model.__call__. If a future mlx-lm release changes
the model architecture, this will need updating. The smoke test
(scripts/smoke_test_mlx.py) validates that the hook works.
"""

from __future__ import annotations

import time
from typing import Optional

from ci_bench.models.base import Model, ModelResponse

# Defer mlx imports so the module can be imported on non-Apple hardware
# (for type checking, testing config loading, etc.). Actual use will
# fail with a clear error if mlx is not installed.
_MLX_AVAILABLE = False
_MLX_LM_VERSION: str | None = None
_MLX_LM_TESTED_RANGE = ("0.20", "0.35")  # Tested with 0.30.x; widen as needed.
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    _MLX_AVAILABLE = True
    try:
        import importlib.metadata
        _MLX_LM_VERSION = importlib.metadata.version("mlx-lm")
    except Exception:
        pass
except ImportError:
    pass


def _check_mlx() -> None:
    if not _MLX_AVAILABLE:
        raise RuntimeError(
            "mlx and mlx-lm are required for MLXModel. "
            "Install with: pip install mlx mlx-lm"
        )


class MLXModel(Model):
    """Wrapper around mlx-lm for inference and activation extraction.

    Usage:
        model = MLXModel("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        response = model.generate("What is the capital of France?")

    For activation extraction, use the extract_hidden_states method
    or the higher-level ci_bench.probes.extract module.
    """

    def __init__(
        self,
        model_path: str,
        max_kv_size: Optional[int] = None,
    ) -> None:
        """Load an mlx-lm model.

        Args:
            model_path: HuggingFace model ID or local path.
                Use mlx-community quantised variants for memory efficiency,
                e.g., "mlx-community/Mistral-7B-Instruct-v0.3-4bit".
            max_kv_size: Optional rotating KV cache size limit.
        """
        _check_mlx()
        if _MLX_LM_VERSION is not None:
            import warnings
            lo, hi = _MLX_LM_TESTED_RANGE
            if not (lo <= _MLX_LM_VERSION < hi):
                warnings.warn(
                    f"mlx-lm {_MLX_LM_VERSION} is outside the tested range "
                    f"({lo}–{hi}). Activation extraction may need updating. "
                    f"Run scripts/smoke_test_mlx.py to verify.",
                    stacklevel=2,
                )
        self._model_path = model_path
        self._model, self._tokenizer = load(model_path)
        self._max_kv_size = max_kv_size

    @property
    def model_id(self) -> str:
        return self._model_path

    @property
    def n_layers(self) -> int:
        """Number of transformer layers."""
        return len(self._model.model.layers)

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension size (from the model config or first layer)."""
        # Most mlx-lm models store args on the model.
        if hasattr(self._model.model, 'args'):
            args = self._model.model.args
            if hasattr(args, 'hidden_size'):
                return args.hidden_size
        # Fallback: inspect the norm layer's weight shape.
        if hasattr(self._model.model, 'norm') and hasattr(self._model.model.norm, 'weight'):
            return self._model.model.norm.weight.shape[0]
        raise AttributeError(
            "Cannot determine hidden_dim. Check model architecture."
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> ModelResponse:
        """Generate a single response.

        Args:
            prompt: The fully rendered prompt string.
            temperature: Sampling temperature (0.0 = greedy).
            max_tokens: Maximum tokens to generate.
            seed: Optional RNG seed for reproducibility. When set,
                mx.random.seed() is called before generation. For
                greedy decoding (temperature=0.0) this has no effect.

        Returns:
            A ModelResponse with full provenance.
        """
        _check_mlx()

        t0 = time.time()

        # Pin mlx RNG for reproducibility when sampling.
        if seed is not None and temperature > 0.0:
            mx.random.seed(seed)

        # Build a sampler appropriate for the temperature.
        if temperature == 0.0:
            sampler = lambda logits: mx.argmax(logits, axis=-1)
        else:
            def sampler(logits):
                return mx.random.categorical(logits / temperature)

        # Use mlx_lm.generate for simplicity. It handles chat templates
        # and tokenisation internally.
        text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )

        elapsed = time.time() - t0

        return ModelResponse(
            text=text,
            model_id=self._model_path,
            prompt_template="",  # Caller fills this in.
            prompt_text=prompt,
            temperature=temperature,
            metadata={"generation_time_s": round(elapsed, 3)},
        )

    def extract_hidden_states(
        self,
        text: str,
    ) -> "mx.array":
        """Extract per-layer hidden states for a single input.

        Runs the model's forward pass once (no generation) and captures
        the hidden state after each transformer layer and after final
        normalization.

        Args:
            text: Input text to tokenise and process.

        Returns:
            mx.array of shape (n_layers + 1, seq_len, hidden_dim).
            Index 0..n_layers-1 are post-layer hidden states.
            Index n_layers is the post-norm hidden state (before the
            language model head).

        Note:
            This method temporarily patches the inner model's __call__
            to intercept layer outputs. It is NOT thread-safe. For batch
            extraction, use ci_bench.probes.extract.extract_batch.
        """
        _check_mlx()

        # Tokenise.
        tokens = self._tokenizer.encode(text)
        input_ids = mx.array([tokens])  # (1, seq_len)

        # We need to intercept the forward pass of model.model (the
        # inner transformer, not the outer LM wrapper). The standard
        # structure is:
        #
        #   x = embed_tokens(input_ids)
        #   for layer in layers:
        #       x = layer(x, mask=mask, cache=cache)
        #   x = norm(x)
        #   logits = lm_head(x)
        #
        # We capture x after each layer and after norm.

        inner_model = self._model.model
        collected_states: list[mx.array] = []

        # Save the original __call__.
        original_call = inner_model.__class__.__call__

        def patched_call(self_inner, x, cache=None, mask=None, **kwargs):
            """Patched forward pass that captures per-layer hidden states."""
            # Embedding.
            if hasattr(self_inner, 'embed_tokens'):
                h = self_inner.embed_tokens(x)
            elif hasattr(self_inner, 'embedding'):
                h = self_inner.embedding(x)
            else:
                raise AttributeError(
                    "Cannot find embedding layer. Expected 'embed_tokens' "
                    "or 'embedding' on the inner model."
                )

            # Create causal mask if needed.
            if mask is None and h.shape[1] > 1:
                mask = nn.MultiHeadAttention.create_additive_causal_mask(
                    h.shape[1]
                )
                mask = mask.astype(h.dtype)

            # Iterate through layers.
            for i, layer in enumerate(self_inner.layers):
                # Different models pass different args to layers.
                # Try the most common signatures.
                try:
                    if cache is not None:
                        try:
                            h = layer(h, mask=mask, cache=cache[i])
                        except TypeError as te:
                            # Only fall back if the error is about the cache
                            # argument, not some other type mismatch.
                            if "cache" in str(te) or "unexpected keyword" in str(te):
                                h = layer(h, mask=mask)
                            else:
                                raise
                    else:
                        h = layer(h, mask=mask)
                except Exception as e:
                    raise RuntimeError(
                        f"Layer {i}/{len(self_inner.layers)} forward "
                        f"pass failed: {e}"
                    ) from e

                # Some layers return (hidden_state, cache_update) tuples.
                if isinstance(h, tuple):
                    h = h[0]

                collected_states.append(h)

            # Final norm.
            h = self_inner.norm(h)
            collected_states.append(h)

            # LM head is on the outer model, not the inner model.
            # Return the normed hidden state so the outer model can
            # apply the head. This matches the expected return type.
            return h

        try:
            # Patch and run.
            inner_model.__class__.__call__ = patched_call
            # Call the outer model (which calls inner_model + lm_head).
            # We don't need the logits; we just need the side effect of
            # populating collected_states.
            _logits = self._model(input_ids)
            mx.eval(_logits)  # Force computation.

            # Stack: (n_layers + 1, seq_len, hidden_dim)
            result = mx.stack(collected_states, axis=0)
            # Remove batch dimension: (n_layers + 1, seq_len, hidden_dim)
            result = result[:, 0, :, :]
            mx.eval(result)

        finally:
            # Always restore the original.
            inner_model.__class__.__call__ = original_call

        return result

    def extract_last_token_hidden_states(
        self,
        text: str,
    ) -> "mx.array":
        """Extract the last-token hidden state from every layer.

        This is the representation used for probing: one vector per layer
        summarising the model's processing of the full input.

        Args:
            text: Input text to tokenise and process.

        Returns:
            mx.array of shape (n_layers + 1, hidden_dim).
        """
        all_states = self.extract_hidden_states(text)
        # Last token along the sequence dimension.
        return all_states[:, -1, :]
