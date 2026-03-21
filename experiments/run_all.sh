#!/bin/bash
# Run the full clean experiment battery.
# Each script loads from existing data (no re-running inference)
# and writes a single canonical JSON to results/canonical/.
#
# Run from repo root: bash experiments/run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "  Clean Experiment Battery"
echo "=========================================="
echo

echo "--- Experiment 1: Behavioral Screening ---"
python3 experiments/exp1_behavioral_screening.py
echo

echo "--- Experiment 2: Activation Probing ---"
python3 experiments/exp2_activation_probing.py
echo

echo "--- Experiment 3: Output-Level Readout ---"
python3 experiments/exp3_output_readout.py
echo

echo "=========================================="
echo "  Battery complete."
echo "  Canonical outputs in results/canonical/"
echo "=========================================="
ls -la results/canonical/experiment_*.json
