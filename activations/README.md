# Activation Data

Hidden-state activations and logit features for the four Tier 1 open-weight models, extracted during CI-Bench evaluation. These files are delivered as GitHub release assets because they are too large for Git.

## Download

```bash
# All assets:
gh release download v1.0-data -D activations/

# Or individually:
gh release download v1.0-data -p "mistral7b.npz" -D activations/
```

Or browse: https://github.com/raeq/propagation-gap/releases/tag/v1.0-data

## Hidden-state activations

| File | Model | Shape | Description |
|---|---|---|---|
| `mistral7b.npz` | Mistral 7B v0.3 | `(338, 33, 4096)` | All 33 layers (L0–L32), 4096-dim |
| `llama8b_full.npz` | Llama 3.1 8B | `(338, 33, 4096)` | All 33 layers (L0–L32), 4096-dim |
| `gemma9b_stratified.npz` | Gemma 2 9B | `(338, 43, 3584)` | All 43 layers (L0–L42), 3584-dim |
| `qwen25_7b.npz` | Qwen2.5 7B | `(338, 29, 3584)` | All 29 layers (L0–L28), 3584-dim |

Arrays are keyed by `"activations"` inside each `.npz` file. Axis ordering: `(questions, layers, hidden_dim)`.

## Logit features

Summary features of the next-token logit distribution, used for the logit-probe control (Section 5.4).

| File | Model | Description |
|---|---|---|
| `logit_features_mistral7b_answer.npz` | Mistral 7B | Answer-token logit vectors |
| `logit_features_mistral7b_confidence.npz` | Mistral 7B | Confidence-token logit vectors |
| `logit_features_llama8b_answer.npz` | Llama 3.1 8B | Answer-token logit vectors |
| `logit_features_llama8b_confidence.npz` | Llama 3.1 8B | Confidence-token logit vectors |
| `logit_features_qwen25_7b_answer.npz` | Qwen2.5 7B | Answer-token logit vectors |
| `logit_features_qwen25_7b_confidence.npz` | Qwen2.5 7B | Confidence-token logit vectors |
| `logit_features_mistral7b.npz` | Mistral 7B | Combined logit features |
| `logit_features_llama8b.npz` | Llama 3.1 8B | Combined logit features |
| `logit_features_gemma9b.npz` | Gemma 2 9B | Combined logit features |

## Loading example

```python
import numpy as np

# Load hidden-state activations
data = np.load("activations/mistral7b.npz")
activations = data["activations"]  # shape: (338, 33, 4096)

# Extract layer 16 (best probing layer for Mistral)
layer_16 = activations[:, 16, :]   # shape: (338, 4096)

# Load logit features
logits = np.load("activations/logit_features_mistral7b_answer.npz")
```

## Correspondence with questions

Row ordering matches the question order in `data/questions/`. Questions are concatenated in construction-type order: K (210), D2 (83), C1 (23), C2 (16), D1 (5), D3 (1) = 338 total.

Model-relative labels (K/D/C per model) are in the V4 summary files under `data/responses/tier1/`.
