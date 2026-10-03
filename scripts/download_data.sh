#!/usr/bin/env bash
# CIC-IDS2017 MachineLearningCVE flow files (Sharafaldin et al., 2018), parquet mirror on Hugging Face.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw
BASE=https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning
for f in Monday-WorkingHours Tuesday-WorkingHours Wednesday-workingHours \
         Thursday-WorkingHours-Morning-WebAttacks Thursday-WorkingHours-Afternoon-Infilteration \
         Friday-WorkingHours-Morning Friday-WorkingHours-Afternoon-PortScan Friday-WorkingHours-Afternoon-DDos; do
  curl -fL -o "data/raw/$f.parquet" "$BASE/$f.pcap_ISCX.csv.parquet"
done
