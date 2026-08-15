#!/usr/bin/env bash
# Qwen3.8-27B (dense hybrid GDN/attention, VL, native MTP) -> MXFP4 for RDNA4 vLLM.
# RTN with olka/qstream on CPU (no GPU). Default exclude list keeps self_attn, mlp.gate,
# lm_head, embed_tokens, visual and mtp in BF16; the 64 MLPs + 48 GDN projections go MXFP4
# (432 Linear). Result 2026-08-15: 55.6 GB -> 22.3 GB, mtp_num_hidden_layers=1 preserved,
# ignore list 505 entries. ~30 min on 32 CPU threads.
set -euo pipefail
VENV=${VENV:-$HOME/quant-venv}
SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)
OUT=${OUT:-$HOME/quant/Qwen3.8-27B-MXFP4}
n=$(ls "$SNAP"*.safetensors | wc -l); [ "$n" -eq 18 ] || { echo "expected 18 shards, found $n" >&2; exit 1; }
"$VENV/bin/qstream-quantize" --model_dir "$SNAP" --output_dir "$OUT" --workers 8 --format ct
echo "done -> $OUT"
