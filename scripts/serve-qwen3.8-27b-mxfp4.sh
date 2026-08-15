#!/usr/bin/env bash
# Serve Qwen3.8-27B-MXFP4 on the RDNA4 vLLM image with native MTP-3.
# Profiles (2026-08-15 measurements, R9700 32 GB):
#   tp2   : 2x R9700, --max-model-len 262144 (native window), 32 slots, KV pool ~345k tokens,
#           ~51 tok/s single-stream (MTP-3 accept ~3.15/step). Needs util 0.95 + batch 8192.
#   single: 1x R9700, --max-model-len 32768, 8 slots, ~41k KV tokens, ~27 tok/s.
#   fp8   : stock Qwen/Qwen3.8-27B-FP8, 2x R9700, --max-model-len 65536, 32 slots (KV pool
#           ~82-126k tokens depending on the profiler run), ~63 tok/s single-stream (MTP-3
#           accept ~3.0/step). Fastest per token; 262k does NOT fit (weights 15.1 GiB/GPU).
#           (64k / 16 slots does NOT fit on one card: the GDN fp32 recurrent state is ~300 MB
#            per slot un-split, plus full-size graphs/activations.)
set -euo pipefail
PROFILE=${1:-tp2}; MODEL=${MODEL:-/quant/Qwen3.8-27B-MXFP4}; PORT=${PORT:-8011}
IMG=${IMG:-capicua25x/vllm-rocm-rdna4:0.26.1-rdna4-rc6}
case "$PROFILE" in
  tp2)    TP=2; MAXLEN=262144; SEQS=32; UTIL=0.95; BATCH=8192 ;;
  single) TP=1; MAXLEN=32768;  SEQS=8;  UTIL=0.95; BATCH=8192 ;;
  fp8)    TP=2; MAXLEN=65536;  SEQS=32; UTIL=0.92; BATCH=16384; MODEL=Qwen/Qwen3.8-27B-FP8 ;;
  *) echo "profile: tp2|single|fp8" >&2; exit 2 ;;
esac
exec docker run --rm --name vllm-qwen38-mxfp4 --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video --group-add=render --ipc=host \
  -e HF_HUB_OFFLINE=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/quant:/quant:ro" \
  --entrypoint /usr/local/bin/vllm "$IMG" \
  serve "$MODEL" --served-model-name qwen --port "$PORT" --trust-remote-code \
  --tensor-parallel-size $TP --gpu-memory-utilization $UTIL --max-model-len $MAXLEN \
  --attention-backend TRITON_ATTN --enable-prefix-caching \
  --max-num-seqs $SEQS --max-num-batched-tokens $BATCH \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
