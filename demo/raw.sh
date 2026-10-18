#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL_ROOT="${MODEL_ROOT:-$ROOT/release_assets/models}"
BOOTSTRAP_PATH="${BOOTSTRAP_PATH:-$ROOT/release_assets/bootstrapped_data/bootstraps.parquet}"

exec python -m dit.infer \
    --model-root "$MODEL_ROOT" \
    --bootstrap-data "$BOOTSTRAP_PATH" \
    --device cuda \
    --display_width 1280 \
    --display_height 960 \
    --sampler ddim \
    --ddim_steps 50 \
    --ddim_eta 0.0 \
    --context_noise_bucket 0 \
    --seed_actions_from_bootstrap \
    "$@"
