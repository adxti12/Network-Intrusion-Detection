#!/usr/bin/env bash
# Reproduce every experiment in the report. Two streams run in parallel on CPU.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=
PY=.venv/bin/python
mkdir -p logs

stream_a() {
  $PY -m src.train --model transformer
  for s in 43 44; do $PY -m src.train --model transformer --seed $s --tag transformer_seed$s; done
}
stream_b() {
  for m in mlp cnn lstm; do $PY -m src.train --model $m; done
  for m in cnn lstm mlp; do $PY -m src.train --model $m --permute-features; done
  for s in 43 44; do for m in mlp cnn lstm; do $PY -m src.train --model $m --seed $s --tag ${m}_seed$s; done; done
}
stream_a > logs/stream_a.log 2>&1 &
stream_b > logs/stream_b.log 2>&1 &
wait
$PY -m src.compare
