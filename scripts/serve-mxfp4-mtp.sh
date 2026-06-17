#!/bin/bash
# SMOKE-TEST launcher for the freshly-quantized DeepSeek-V4-Pro Thinking distill MXFP4.
# Mirrors the prod wrapper (apexia-vllm-qwen.sh) but:
#   - points at the LOCAL quant dir (not the HF cache model)
#   - MTP-1 enabled: base Qwen3.6 MTP block grafted in (graft-mtp.py); mtp_num_hidden_layers=1
#   - max-model-len 32768 (fast load / small KV for a smoke test; base supports 262144)
#   - container name apexia-vllm-try so it won't collide with the prod unit's name
# Serves on :8011 as "qwen" so the existing test harness / curl probes just work.
# PROD apexia-vllm-qwen MUST be stopped first (both cards are needed; can't co-fit).
set -euo pipefail

MODEL=$HOME/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4

R1=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render)
C1=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-card)
R2=$(readlink -f /dev/dri/by-path/pci-0000:06:00.0-render)
C2=$(readlink -f /dev/dri/by-path/pci-0000:06:00.0-card)
for d in "$R1" "$C1" "$R2" "$C2" /dev/kfd; do
  [ -e "$d" ] || { echo "required device $d not found" >&2; exit 1; }
done

exec docker run --rm --name apexia-vllm-try --network=host \
  --device=/dev/kfd --device="$R1" --device="$C1" --device="$R2" --device="$C2" \
  --group-add=video --group-add=render --ipc=host --security-opt seccomp=unconfined \
  -e HF_HUB_OFFLINE=1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v "$MODEL":"$MODEL":ro \
  tcclaviger/vllm-rocm-mxfp4-nvfp4:latest \
  "$MODEL" \
  --served-model-name qwen --port 8011 --trust-remote-code \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 32768 \
  --enable-prefix-caching --max-num-seqs 64 \
  --language-model-only \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
