# The Propagation Gap

Companion repository for:

> **The Propagation Gap: Hidden-State Reliability Signals Do Not Reach Language Model Outputs**
>
> Richard Quinn (2026). Submitted to *Transactions on Machine Learning Research*.

## Abstract

Large language models maintain internal representations that distinguish known from unknown material, yet their outputs fail to reflect this distinction. We introduce CI-Bench, a 338-question factual benchmark with model-relative labels (Known, Declined, Contested) derived from each model's own response patterns. Linear probes trained on hidden-state activations from three open-weight models (Mistral 7B, Llama 3.1 8B, Gemma 2 9B) classify Known versus Declined items with AUROC 0.74--0.76 at the best intermediate layer. Performance degrades monotonically toward the output layer (AUROC 0.67--0.70) and collapses further in logit-derived and verbalized-confidence readouts (AUROC 0.56--0.63). Bootstrap paired-gap tests confirm the best-layer-to-last-layer drop and the best-layer-to-behavioral gap are statistically reliable. Surface controls (TF-IDF, answer-string features, grouped cross-validation) rule out lexical confounds. We term this pattern the *propagation gap*: a measurable, layer-by-layer loss of reliability information between hidden representations and model outputs.

## Repository structure

```
propagation-gap/
├── data/
│   ├── questions/          # CI-Bench v1.0 (338 questions, 6 construction types)
│   └── responses/          # Per-model responses and V4 summaries
│       ├── tier1/          #   Mistral 7B, Llama 3.1 8B, Gemma 2 9B
│       └── tier2/          #   GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet
├── results/
│   ├── canonical/          # Canonical experiment summaries (Experiments 1--3)
│   ├── probes/             # Per-model probe results (original labels)
│   ├── v4_probes.json      # V4 model-relative probe results
│   ├── v4_layer_trajectory.json  # Per-layer AUROC trajectory
│   ├── bootstrap_gaps.json # Bootstrap paired-gap tests (N=5000)
│   ├── surface_controls.json    # TF-IDF, surface features, grouped CV
│   └── logit_probe.json   # Logit-derived probe results
├── activations/            # Hidden-state activations (see activations/README.md)
├── experiments/            # Runnable experiment scripts
├── analysis/               # Jupyter notebooks reproducing all tables and figures
├── ci_bench/               # Python package (benchmark framework)
├── configs/                # Experiment configuration files
└── tests/                  # Unit tests
```

## Quick start

```bash
# Clone and install
git clone https://github.com/raeq/propagation-gap.git
cd propagation-gap
pip install -e .

# Reproduce analyses from released data
jupyter notebook analysis/01_behavioral_screening.ipynb
```

To rerun Tier 1 experiments (requires local GPU with MLX):
```bash
pip install -e ".[mlx]"
bash experiments/run_all.sh
```

To rerun Tier 2 experiments (requires API keys):
```bash
pip install -e ".[api]"
cp .env.example .env
# Edit .env with your API keys
python experiments/exp1_behavioral_screening.py --tier2
```

## Activation data

Hidden-state activation files (~235 MB compressed) are too large for Git. They are available as a GitHub release artifact:

**[Download activations](https://github.com/raeq/propagation-gap/releases/tag/v1.0-data)**

See `activations/README.md` for file format, array shapes, and loading instructions.

## Citation

```bibtex
@article{quinn2026propagation,
  title={The Propagation Gap: Hidden-State Reliability Signals
         Do Not Reach Language Model Outputs},
  author={Quinn, Richard},
  journal={Transactions on Machine Learning Research},
  year={2026},
  note={Under review}
}
```

## License

MIT. See [LICENSE](LICENSE).
