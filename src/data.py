"""CIC-IDS2017 loading, cleaning, label grouping, splitting and scaling.

Run as a script to build the processed dataset once:
    python -m src.data
which writes data/processed/{train,val,test}.npz, meta.json and EDA figures.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (BENIGN, CLASS_NAMES, DATA_PROCESSED, DATA_RAW, FIGURES, SEED,
                        TRAIN_CAP_PER_CLASS, VAL_BENIGN_CAP)

# Raw label -> attack family. Infiltration (36 flows) and Heartbleed (11 flows)
# are dropped: too few samples to learn from or to evaluate meaningfully.
LABEL_MAP = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS", "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Web Attack � Brute Force": "WebAttack",
    "Web Attack � XSS": "WebAttack",
    "Web Attack � Sql Injection": "WebAttack",
    "Bot": "Bot",
}

# Destination Port identifies the service rather than describing flow
# behaviour; models can shortcut on it (e.g. port 21/22 => Patator), so it is
# excluded to keep the comparison about learning from traffic statistics.
DROP_COLUMNS = ["Destination Port", "Fwd Header Length_duplicated_0"]


def load_raw() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_RAW, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_RAW}; run scripts/download_data.sh")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


def clean(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    log["raw_rows"] = len(df)
    log["raw_label_counts"] = df["Label"].value_counts().to_dict()

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    feats = df.columns.drop("Label")
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    n = len(df)
    df = df.dropna()
    log["dropped_nan_inf"] = n - len(df)

    n = len(df)
    df = df.drop_duplicates()
    log["dropped_duplicates"] = n - len(df)

    constant = [c for c in feats if df[c].nunique() <= 1]
    df = df.drop(columns=constant)
    log["dropped_constant_columns"] = constant

    n = len(df)
    df = df[df["Label"].isin(LABEL_MAP)].copy()
    log["dropped_rare_classes"] = n - len(df)
    df["Class"] = df["Label"].map(LABEL_MAP)
    log["clean_rows"] = len(df)
    log["class_counts"] = df["Class"].value_counts().to_dict()
    return df


def signed_log1p(x: np.ndarray) -> np.ndarray:
    # Flow features span many orders of magnitude (bytes/s up to ~1e9) and a
    # few contain negatives (e.g. Init_Win_bytes = -1), so compress symmetrically.
    return np.sign(x) * np.log1p(np.abs(x))


def cap_per_class(X, y, cap, rng):
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    keep = rng.permutation(np.concatenate(keep))
    return X[keep], y[keep]


def eda_figures(df: pd.DataFrame, feature_names, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(FIGURES, exist_ok=True)

    raw = pd.Series(log["raw_label_counts"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    raw.plot.barh(ax=ax, logx=True, color="#4C72B0")
    ax.set_xlabel("Flows (log scale)")
    ax.set_title("CIC-IDS2017 raw label distribution")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "eda_raw_labels.png"), dpi=150); plt.close(fig)

    counts = df["Class"].value_counts().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot.bar(ax=ax, logy=True, color="#55A868")
    ax.set_ylabel("Flows (log scale)")
    ax.set_title("Class distribution after cleaning and grouping")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "eda_class_counts.png"), dpi=150); plt.close(fig)

    sample = df.sample(min(len(df), 200_000), random_state=SEED)
    corr = sample[feature_names].corr().abs()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="viridis", ax=ax, xticklabels=False, yticklabels=False,
                cbar_kws={"label": "|Pearson r|"})
    ax.set_title("Absolute feature correlation (features in dataset column order)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "eda_feature_correlation.png"), dpi=150); plt.close(fig)

    # Share of highly correlated pairs among adjacent vs. random feature pairs:
    # quantifies how much "locality" exists in the column order for the CNN.
    c = corr.values
    adj = np.array([c[i, i + 1] for i in range(len(c) - 1)])
    iu = np.triu_indices(len(c), 2)
    log["mean_abs_corr_adjacent_features"] = float(np.nanmean(adj))
    log["mean_abs_corr_nonadjacent_features"] = float(np.nanmean(c[iu]))


def build():
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    log = {}
    df = clean(load_raw(), log)
    feature_names = [c for c in df.columns if c not in ("Label", "Class")]
    log["n_features"] = len(feature_names)
    eda_figures(df, feature_names, log)

    X = signed_log1p(df[feature_names].to_numpy(dtype=np.float64)).astype(np.float32)
    y = df["Class"].map({c: i for i, c in enumerate(CLASS_NAMES)}).to_numpy(np.int64)
    del df

    # 70/15/15 stratified split on the full cleaned data. Validation and test
    # keep the natural (imbalanced) distribution so metrics reflect deployment.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    log["split_full_train"] = {CLASS_NAMES[c]: int(n) for c, n in zip(*np.unique(y_tr, return_counts=True))}

    # Cap majority classes in the training set only (CPU budget + imbalance).
    X_tr, y_tr = cap_per_class(X_tr, y_tr, TRAIN_CAP_PER_CLASS, np.random.default_rng(SEED))
    # Early-stopping set: all validation attacks, benign subsampled (CPU budget).
    rng = np.random.default_rng(SEED + 1)
    benign = np.flatnonzero(y_va == BENIGN)
    keep = np.sort(np.concatenate([np.flatnonzero(y_va != BENIGN),
                                   rng.choice(benign, min(VAL_BENIGN_CAP, len(benign)), replace=False)]))
    # Benign weight restores the natural class ratio when scoring validation F1.
    log["val_benign_weight"] = len(benign) / min(VAL_BENIGN_CAP, len(benign))
    X_va, y_va = X_va[keep], y_va[keep]

    scaler = StandardScaler().fit(X_tr)
    for name, Xs, ys in (("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)):
        Xs = np.clip(scaler.transform(Xs), -10, 10).astype(np.float32)
        np.savez_compressed(os.path.join(DATA_PROCESSED, f"{name}.npz"), X=Xs, y=ys)
        log[f"split_{name}"] = {CLASS_NAMES[c]: int(n) for c, n in zip(*np.unique(ys, return_counts=True))}

    log["feature_names"] = feature_names
    log["class_names"] = CLASS_NAMES
    with open(os.path.join(DATA_PROCESSED, "meta.json"), "w") as f:
        json.dump(log, f, indent=2, default=int)
    print(json.dumps({k: v for k, v in log.items() if k != "feature_names"}, indent=2, default=int))


def load_split(name: str):
    d = np.load(os.path.join(DATA_PROCESSED, f"{name}.npz"))
    return d["X"], d["y"]


def load_meta() -> dict:
    with open(os.path.join(DATA_PROCESSED, "meta.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    build()
