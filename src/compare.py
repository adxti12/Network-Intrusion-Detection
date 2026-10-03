"""Aggregate all runs into comparison tables, figures and significance tests.

    python -m src.compare

Writes results/summary.md, results/summary.json and results/figures/*.png.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import chi2
from sklearn.metrics import average_precision_score

from src import config as C
from src.data import load_meta, load_split
from src.models import DISPLAY, FTTransformer

MODELS = ["mlp", "cnn", "lstm", "transformer"]
COLORS = {"mlp": "#4C72B0", "cnn": "#DD8452", "lstm": "#55A868", "transformer": "#C44E52"}


def load_runs():
    runs = {}
    for f in glob.glob(os.path.join(C.RESULTS, "*", "metrics.json")):
        with open(f) as fh:
            m = json.load(fh)
        runs[m["run"]] = m
    return runs


def fmt(x, d=4):
    return f"{x:.{d}f}"


def main_table(runs):
    rows = []
    for k in MODELS:
        if k not in runs:
            continue
        m = runs[k]
        b = m["binary"]
        rows.append({
            "Model": DISPLAY[k], "Params": f"{m['n_params']:,}",
            "Accuracy": m["accuracy"], "Macro P": m["macro_precision"], "Macro R": m["macro_recall"],
            "Macro F1": m["macro_f1"], "Weighted F1": m["weighted_f1"],
            "Attack P": b["attack"]["precision"], "Attack R": b["attack"]["recall"],
            "Attack F1": b["attack"]["f1"], "Benign F1": b["benign"]["f1"],
            "FPR": b["false_positive_rate"], "FNR": b["false_negative_rate"],
            "Epochs (best)": f"{m['epochs_trained']} ({m['best_epoch']})",
            "Train time (min)": m["train_seconds"] / 60, "Inference (µs/flow)": m["inference_us_per_flow"],
        })
    return pd.DataFrame(rows)


def per_class_table(runs):
    data = {DISPLAY[k]: [runs[k]["per_class"][c]["f1-score"] for c in C.CLASS_NAMES] for k in MODELS if k in runs}
    df = pd.DataFrame(data, index=C.CLASS_NAMES)
    df["Support"] = [int(runs[next(k for k in MODELS if k in runs)]["per_class"][c]["support"]) for c in C.CLASS_NAMES]
    return df


def per_class_recall(runs):
    return pd.DataFrame({DISPLAY[k]: [runs[k]["per_class"][c]["recall"] for c in C.CLASS_NAMES]
                         for k in MODELS if k in runs}, index=C.CLASS_NAMES)


def seed_table(runs):
    rows = []
    for k in MODELS:
        ids = [r for r in [k, f"{k}_seed43", f"{k}_seed44"] if r in runs]
        if len(ids) < 2:
            continue
        f1 = np.array([runs[r]["macro_f1"] for r in ids])
        acc = np.array([runs[r]["accuracy"] for r in ids])
        rec = np.array([runs[r]["binary"]["attack"]["recall"] for r in ids])
        rows.append({"Model": DISPLAY[k], "Seeds": len(ids),
                     "Macro F1": f"{f1.mean():.4f} ± {f1.std(ddof=1):.4f}",
                     "Accuracy": f"{acc.mean():.4f} ± {acc.std(ddof=1):.4f}",
                     "Attack recall": f"{rec.mean():.4f} ± {rec.std(ddof=1):.4f}"})
    return pd.DataFrame(rows)


def ablation_table(runs):
    rows = []
    for k in MODELS:
        orig = [runs[r] for r in [k, f"{k}_seed43", f"{k}_seed44"] if r in runs]
        perm = [runs[r] for r in [f"{k}_permuted", f"{k}_permuted_seed43", f"{k}_permuted_seed44"] if r in runs]
        if orig and perm:
            fo, fp = np.mean([m["macro_f1"] for m in orig]), np.mean([m["macro_f1"] for m in perm])
            rows.append({"Model": DISPLAY[k], "Runs (orig/perm)": f"{len(orig)}/{len(perm)}",
                         "Macro F1 (original order)": fo, "Macro F1 (permuted order)": fp, "Δ Macro F1": fp - fo,
                         "Accuracy (original)": np.mean([m["accuracy"] for m in orig]),
                         "Accuracy (permuted)": np.mean([m["accuracy"] for m in perm])})
    return pd.DataFrame(rows)


def pr_auc_table(runs):
    """Threshold-free view: per-class average precision (area under PR curve)."""
    data = {}
    for k in MODELS:
        p = os.path.join(C.RESULTS, k, "test_predictions.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        proba = d["proba"].astype(np.float32)
        ap = [average_precision_score(d["y_true"] == i, proba[:, i]) for i in range(len(C.CLASS_NAMES))]
        attack_score = 1.0 - proba[:, C.BENIGN]
        data[DISPLAY[k]] = ap + [float(np.mean(ap)),
                                 average_precision_score(d["y_true"] != C.BENIGN, attack_score)]
    return pd.DataFrame(data, index=C.CLASS_NAMES + ["Macro AP", "Attack-vs-benign AP"])


def mcnemar(runs):
    """Pairwise McNemar tests (with continuity correction) on test-set correctness."""
    preds = {}
    for k in MODELS:
        p = os.path.join(C.RESULTS, k, "test_predictions.npz")
        if os.path.exists(p):
            d = np.load(p)
            preds[k] = d["y_pred"] == d["y_true"]
    rows = []
    keys = list(preds)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = preds[keys[i]], preds[keys[j]]
            n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
            stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0.0
            rows.append({"A": DISPLAY[keys[i]], "B": DISPLAY[keys[j]],
                         "A right, B wrong": n01, "B right, A wrong": n10,
                         "χ²": round(stat, 2), "p-value": f"{chi2.sf(stat, 1):.2e}"})
    return pd.DataFrame(rows)


def plot_confusions(runs):
    present = [k for k in MODELS if k in runs]
    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.8))
    for ax, k in zip(np.atleast_1d(axes), present):
        cm = np.array(runs[k]["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True)
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, annot_kws={"size": 7})
        ax.set_title(f"{DISPLAY[k]} (row-normalised)")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "confusion_matrices.png"), dpi=150); plt.close(fig)


def plot_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for k in MODELS:
        p = os.path.join(C.RESULTS, k, "history.csv")
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p)
        axes[0].plot(h.epoch, h.train_loss, label=DISPLAY[k], color=COLORS[k])
        axes[1].plot(h.epoch, h.val_macro_f1, label=DISPLAY[k], color=COLORS[k], marker="o", ms=3)
    axes[0].set(title="Training loss (class-weighted CE)", xlabel="Epoch", yscale="log")
    axes[1].set(title="Validation macro-F1", xlabel="Epoch")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "training_curves.png"), dpi=150); plt.close(fig)


def plot_per_class(pc):
    df = pc.drop(columns="Support")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.5, vmax=1.0, ax=ax)
    ax.set_yticklabels([f"{c} (n={s:,})" for c, s in zip(pc.index, pc.Support)], rotation=0)
    ax.set_title("Per-class F1 on the test set")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "per_class_f1.png"), dpi=150); plt.close(fig)


def plot_summary_bars(mt):
    metrics = ["Accuracy", "Macro F1", "Attack R", "Attack P"]
    x = np.arange(len(metrics)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (_, r) in enumerate(mt.iterrows()):
        k = [m for m in MODELS if DISPLAY[m] == r.Model][0]
        ax.bar(x + (i - 1.5) * w, [r[m] for m in metrics], w, label=r.Model, color=COLORS[k])
    ax.set_xticks(x, ["Accuracy", "Macro F1 (7 classes)", "Attack recall", "Attack precision"])
    lo = min(mt[metrics].min()) - 0.03
    ax.set_ylim(max(0, lo), 1.0); ax.legend(ncol=4, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Test-set performance")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "model_comparison.png"), dpi=150); plt.close(fig)


def plot_ablation(ab):
    if ab.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ab)); w = 0.35
    ax.bar(x - w / 2, ab["Macro F1 (original order)"], w, label="Original column order", color="#4C72B0")
    ax.bar(x + w / 2, ab["Macro F1 (permuted order)"], w, label="Randomly permuted columns", color="#DD8452")
    ax.set_xticks(x, ab.Model); ax.set_ylabel("Test macro-F1")
    lo = min(ab["Macro F1 (original order)"].min(), ab["Macro F1 (permuted order)"].min())
    ax.set_ylim(max(0, lo - 0.05), 1.0); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Does feature order matter? (ablation)")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "feature_order_ablation.png"), dpi=150); plt.close(fig)


def transformer_attention(meta, n=20_000):
    ckpt = os.path.join(C.CHECKPOINTS, "transformer.pt")
    if not os.path.exists(ckpt):
        return None
    X, y = load_split("test")
    rng = np.random.default_rng(C.SEED)
    model = FTTransformer(X.shape[1], len(C.CLASS_NAMES))
    model.load_state_dict(torch.load(ckpt)); model.eval()
    names = meta["feature_names"]
    per_class = {}
    for ci, cname in enumerate(C.CLASS_NAMES):
        idx = np.flatnonzero(y == ci)
        idx = rng.choice(idx, min(len(idx), n // len(C.CLASS_NAMES)), replace=False)
        att = model.cls_attention(torch.from_numpy(X[idx]))[-1].numpy()  # last layer
        per_class[cname] = att
    df = pd.DataFrame(per_class, index=names)
    top = df.mean(1).sort_values(ascending=False).index[:15]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df.loc[top], cmap="magma", ax=ax, annot=True, fmt=".3f", annot_kws={"size": 7})
    ax.set_title("Transformer: last-layer [CLS] attention, top-15 features, by true class")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIGURES, "transformer_attention.png"), dpi=150); plt.close(fig)
    return df.loc[top]


def top_confusions(runs, k=3):
    out = {}
    for m in MODELS:
        if m not in runs:
            continue
        cm = np.array(runs[m]["confusion_matrix"])
        np.fill_diagonal(cm, 0)
        flat = np.argsort(cm.ravel())[::-1][:k]
        out[DISPLAY[m]] = [f"{C.CLASS_NAMES[i // cm.shape[1]]} → {C.CLASS_NAMES[i % cm.shape[1]]}: {cm.ravel()[i]}"
                           for i in flat if cm.ravel()[i] > 0]
    return out


def md(df, floatfmt=4):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: f"{v:.{floatfmt}f}")
    return df.to_markdown(index=False)


def main():
    os.makedirs(C.FIGURES, exist_ok=True)
    runs, meta = load_runs(), load_meta()
    mt, pc, pr = main_table(runs), per_class_table(runs), per_class_recall(runs)
    st, ab, mc = seed_table(runs), ablation_table(runs), mcnemar(runs)
    ap = pr_auc_table(runs)

    plot_confusions(runs); plot_curves(); plot_per_class(pc); plot_summary_bars(mt); plot_ablation(ab)
    att = transformer_attention(meta)
    tc = top_confusions(runs)

    lines = ["# Results summary (auto-generated by `python -m src.compare`)", "",
             f"Test set: {sum(meta['split_test'].values()):,} flows "
             f"({meta['split_test']['Benign']:,} benign). Seed {C.SEED} unless stated.", "",
             "## Overall test performance", "", md(mt), "",
             "## Per-class F1", "", pc.reset_index(names="Class").to_markdown(index=False, floatfmt=".4f"), "",
             "## Per-class recall", "", pr.reset_index(names="Class").to_markdown(index=False, floatfmt=".4f"), "",
             "## Average precision (PR-AUC, threshold-free)", "",
             ap.reset_index(names="Class").to_markdown(index=False, floatfmt=".4f"), "",
             "## Most frequent confusions (true → predicted: count)", ""]
    for m, cs in tc.items():
        lines.append(f"- **{m}**: " + "; ".join(cs))
    if not st.empty:
        lines += ["", "## Seed variance (mean ± std over seeds 42/43/44)", "", st.to_markdown(index=False)]
    if not ab.empty:
        lines += ["", "## Feature-order ablation", "", md(ab)]
    if not mc.empty:
        lines += ["", "## Pairwise McNemar tests (test-set correctness)", "", mc.to_markdown(index=False)]
    if att is not None:
        lines += ["", "## Transformer [CLS] attention — top features (last layer)", "",
                  att.reset_index(names="Feature").to_markdown(index=False, floatfmt=".3f")]
    with open(os.path.join(C.RESULTS, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(C.RESULTS, "summary.json"), "w") as f:
        json.dump({"main": mt.to_dict("records"), "per_class_f1": pc.to_dict(), "seeds": st.to_dict("records"),
                   "ablation": ab.to_dict("records"), "average_precision": ap.to_dict(), "mcnemar": mc.to_dict("records"),
                   "top_confusions": tc}, f, indent=2, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
