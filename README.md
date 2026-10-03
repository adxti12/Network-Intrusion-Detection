# Comparative Deep Learning Approaches for Network Intrusion Detection

ICT-4442 Deep Learning mini project — Aditi Arun (230911108), Anushka Thakur (230911048),
School of Computer Engineering, MIT Manipal.

Four deep learning architecture families — **MLP**, **1D CNN**, **LSTM** and a
**Transformer** (feature-tokenizer / FT-Transformer) — are trained and evaluated on
**CIC-IDS2017** under one shared experimental protocol, so that performance
differences can be attributed to the architecture. Results and analysis are in
[`REPORT.md`](REPORT.md); auto-generated tables are in
[`results/summary.md`](results/summary.md).

## Repository layout

```
src/
  config.py    shared protocol: paths, seed, classes, split caps, optimiser settings
  data.py      load → clean → group labels → split → scale; EDA figures
  models.py    MLP, CNN1D, LSTMNet, FTTransformer
  train.py     common training loop, early stopping, test evaluation, metrics
  compare.py   comparison tables, figures, McNemar tests, attention analysis
scripts/
  download_data.sh   fetch the CIC-IDS2017 flow files (~270 MB parquet)
  run_all.sh         every experiment in the report, then src.compare
results/
  <run>/metrics.json, history.csv   one directory per training run
  figures/                          all plots used in the report
  summary.md / summary.json         aggregated results
```

## Reproducing

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

./scripts/download_data.sh          # data/raw/*.parquet
.venv/bin/python -m src.data        # data/processed/{train,val,test}.npz + EDA figures
./scripts/run_all.sh                # all training runs + src.compare (CPU, a few hours)
```

Train a single model: `python -m src.train --model {mlp,cnn,lstm,transformer} [--permute-features] [--seed N]`.

> If your shell exports a `PYTHONPATH` (e.g. ROS), clear it (`PYTHONPATH= python ...`)
> so the venv's packages are used.

## Experimental protocol (identical for every model)

| Item | Setting |
|---|---|
| Data | CIC-IDS2017 MachineLearningCVE flows, all 8 day files (2,830,743 flows) |
| Cleaning | drop NaN/±inf rows, exact duplicates, constant columns, duplicated header column, `Destination Port` |
| Classes | 7: Benign, DoS, DDoS, PortScan, BruteForce (FTP/SSH-Patator), WebAttack, Bot (Infiltration and Heartbleed dropped: 36 and 11 flows) |
| Features | 68 numeric flow statistics, signed `log1p`, standardised on train, clipped to ±10 |
| Split | stratified 70 / 15 / 15, seed 42 |
| Train set | majority classes capped at 60k flows each (190,280 flows) |
| Validation | all attacks + 60k benign; benign re-weighted to the natural ratio when scoring |
| Test | full, untouched natural distribution (334,968 flows, 85% benign) |
| Loss | cross-entropy with inverse-sqrt class weights |
| Optimiser | AdamW, lr 1e-3, wd 1e-4, batch 512, grad-clip 1.0, ReduceLROnPlateau(×0.5) |
| Stopping | best validation macro-F1, patience 6, max 30 epochs |
| Metrics | accuracy, macro/weighted P/R/F1, per-class report, benign-vs-attack P/R/F1, FPR/FNR |
