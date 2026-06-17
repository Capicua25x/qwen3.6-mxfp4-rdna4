#!/usr/bin/env bash
# Quantize nerkyor/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill (BF16 DeepSeek-V4-Pro
# thinking-style distill, newest Lynn release 2026-06-07)
# -> MXFP4, replicating the EXACT recipe + tool of the working
#    pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4 so it loads on the same
#    tcclaviger RDNA4 vLLM container that prod runs.
#
# Tool: olka/qstream @ master (incl. merged PR #1 "Fix MXFP4 quantization of
#       Qwen3_5-MoE-family models (incl. Qwen3.6)"). PR #1 is what pahajoki used;
#       its end-to-end verification was run on THIS hardware (2x R9700, tcclaviger).
#
# Recipe = qstream-quantize DEFAULTS (PR #1 made the defaults correct):
#   - format ct            (compressed-tensors, per-expert .weight_packed/.weight_scale)
#   - MSE-optimal scale select (3 candidates/block) ............. ON  (default)
#   - activation-aware gamma proxy (input_layernorm.weight) ..... ON  (default, no calib)
#   - exclude (kept BF16): *self_attn* *.mlp.gate. *shared_expert*
#                          *lm_head* *embed_tokens* *visual* *mtp*
#     => quantizes ONLY linear_attn projections + the 256 routed experts.
#        vision tower, MTP draft, shared expert, routing gate, attn stay BF16
#        (this is the whole reason the pre-made NVFP4-v8-RTN crashed on our
#         container: it quantized the vision tower -> qwen3_vl KeyError).
#
# RTN / data-free: no GPU needed. Runs on CPU so it does NOT touch the two
# R9700s that prod vLLM is serving on.
set -euo pipefail

VENV=/home/ludwid/quant-venv
SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--nerkyor--Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill/snapshots/*/ | head -1)
OUT=/home/ludwid/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4

echo "model_dir : $SNAP"
echo "output_dir: $OUT"

# sanity: all 16 shards + index present before we start
n=$(ls "$SNAP"*.safetensors 2>/dev/null | wc -l)
[ -f "$SNAP/model.safetensors.index.json" ] || { echo "FATAL: index.json missing (download incomplete)"; exit 1; }
[ "$n" -eq 16 ] || { echo "FATAL: expected 16 shards, found $n (download incomplete)"; exit 1; }

"$VENV/bin/qstream-quantize" \
    --model_dir "$SNAP" \
    --output_dir "$OUT" \
    --workers 8 \
    --format ct
# all other knobs left at default == pahajoki recipe
