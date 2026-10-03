"""Shared experimental protocol: every model is trained and evaluated with these settings."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
CHECKPOINTS = os.path.join(RESULTS, "checkpoints")

SEED = 42
CLASS_NAMES = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot"]
BENIGN = 0

TRAIN_CAP_PER_CLASS = 60_000
VAL_BENIGN_CAP = 60_000  # early-stopping subsample; test set is never subsampled

# Training protocol (identical for all architectures)
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096
MAX_EPOCHS = 30
PATIENCE = 6            # early stopping on validation macro-F1
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0
