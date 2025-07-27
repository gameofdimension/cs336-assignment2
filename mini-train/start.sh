set -xeuo pipefail

ts=$(date +%s)
LOG_PATH="cs336-train-${ts}.log"

uv run python \
    mini-train/train.py \
    --vocab_size=10000 \
    --d_model=512 \
    --d_ff=1344 \
    --num_layers=4 \
    --num_heads=16 \
    --context_length=256 \
    --dataset_path=$1 \
    --batch_size=$2 2>&1 | tee -a $LOG_PATH