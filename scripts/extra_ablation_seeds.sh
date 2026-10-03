#!/usr/bin/env bash
# Extra seeds for the permuted-feature CNN, so the ablation is not a single-seed comparison.
cd "$(dirname "$0")/.."
export PYTHONPATH=
for s in 43 44; do .venv/bin/python -m src.train --model cnn --permute-features --seed $s --tag cnn_permuted_seed$s; done
