#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

RAW_META="${1:-data/amazon_beauty/meta_Beauty.json}"
FILTERED_META="data/amazon_beauty/meta_Beauty.filtered.jsonl"

python new_pipe/prepare_beauty_meta.py \
  --raw-meta "$RAW_META" \
  --metadata-csv data/amazon_beauty/metadata.csv \
  --output "$FILTERED_META"

python new_pipe/run_beauty_unified_eval_pipeline.py \
  --query-csv data/amazon_beauty/query_data1.csv \
  --filtered-meta-jsonl "$FILTERED_META" \
  --topk 200 \
  --fallback-topk 500
