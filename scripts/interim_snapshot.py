"""Reproduce the Interim-report snapshot: the initial, unoptimised pipeline.

Initial settings (before tuning): training classes capped at 120k, full
unweighted validation set, and a single training epoch for the MLP and 1D CNN.
Outputs go to data/processed_interim/ and results_interim/ so the final
experiments are untouched.

    PYTHONPATH= .venv/bin/python scripts/interim_snapshot.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C  # noqa: E402

C.DATA_PROCESSED = os.path.join(C.ROOT, "data", "processed_interim")
C.RESULTS = os.path.join(C.ROOT, "results_interim")
C.FIGURES = os.path.join(C.RESULTS, "figures")
C.CHECKPOINTS = os.path.join(C.RESULTS, "checkpoints")
C.TRAIN_CAP_PER_CLASS = 120_000
C.VAL_BENIGN_CAP = 10 ** 9   # no validation subsampling => unweighted validation

from src import data, train  # noqa: E402  (must import after patching config)

if __name__ == "__main__":
    if not os.path.exists(os.path.join(C.DATA_PROCESSED, "meta.json")):
        data.build()
    for m in ("mlp", "cnn"):
        train.run(m, max_epochs=1)
