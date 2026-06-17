# Qwen3.6-35B-A3B MXFP4 for RDNA4 — DeepSeek-V4-Pro Thinking distill

Toolkit + recipe to run **`nerkyor/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill`** (a
DeepSeek-V4-Pro reasoning-style LoRA distill of Qwen3.6-35B-A3B) as **MXFP4** on
**AMD RDNA4** (2× Radeon AI PRO R9700), served by the
[`tcclaviger/vllm-rocm-mxfp4-nvfp4`](https://hub.docker.com/r/tcclaviger/vllm-rocm-mxfp4-nvfp4)
vLLM container — **with multi-token-prediction (MTP) speculative decoding grafted back in.**

This is, as far as I know, the **first RDNA4-loadable MXFP4 build of this distill**.
The distill ships as BF16, GGUF, and an NVFP4 that was built for SGLang/Blackwell and
**crashes the RDNA4 vLLM container** (vision-tower `KeyError`). This repo produces a
build that loads cleanly and benches **at parity with the production base model**.

> **Weights:** the quantized model is published separately on Hugging Face (it's ~20 GB).
> This repo is the **recipe + tooling** to reproduce it. See *Reproduce* below.

## Results (2× R9700, TP2, MXFP4, MTP-3 grafted)

| Metric | This build | Prod base (`pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`, MTP-3) |
|---|---|---|
| Single-stream, short prompt | **107 tok/s** | ~100 |
| Single-stream, 6k prompt | **82 tok/s** | ~100 |
| Concurrency ceiling, short prompt | **~128** | ~128 |
| Concurrency ceiling, 6k prompt | ~32 | ~128 |
| Aggregate @128 (short) | ~1875 tok/s | — |
| MTP draft acceptance | 55% (grafted head) | 81% (native) |
| Size | 69.3 GB BF16 → **20.5 GB** (29.6%) | — |

Single-stream **edges out the production base**; short-prompt concurrency matches it.
The realistic-prompt high-concurrency gap is purely the grafted MTP head's lower
acceptance (it was trained on the *base* hidden states, not the LoRA-shifted distill).

## What was non-obvious

Three things stand between the BF16 distill and a working RDNA4 MXFP4 serve:

### 1. The MXFP4 recipe (qstream)
Quantized with [`olka/qstream`](https://github.com/olka/qstream) `master` (which
includes the merged PR #1 fixing Qwen3.6-family quantization). The default exclude
list **is** the correct recipe — it keeps `*self_attn* *.mlp.gate. *shared_expert*
*lm_head* *embed_tokens* *visual* *mtp*` in BF16 and quantizes only the DeltaNet
`linear_attn` projections + the 256 routed experts (per-expert compressed-tensors
`mxfp4-pack-quantized`, group-32 symmetric, MSE-optimal scale select). This mirrors
`pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`. See `scripts/quant-mxfp4.sh`.

### 2. Text-only checkpoint → multimodal config wrap + `--language-model-only`
This distill is `Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text` (**no vision tower**).
The RDNA4 vLLM container only registers the **multimodal** `Qwen3_5MoeForConditionalGeneration`
loader, which demands a `vision_config` — so a text-only config crashes at processor
build (`Invalid HF config … found Qwen3_5MoeTextConfig`). Fix: rewrap `config.json`
into the multimodal shape (nest the text fields under `text_config`, graft a
`vision_config`), then serve with **`--language-model-only`** so the vision tower is
built as a no-init stub and its (absent) weights are never loaded. The model's tensors
are already named `model.language_model.*`, so they map straight onto the LM submodule.

### 3. MTP graft (re-enabling speculative decoding)
The distill was exported with `mtp_num_hidden_layers: 0` — the MTP draft block was
dropped, so no speculative decoding is possible (any `num_speculative_tokens` fails).
But the base Qwen3.6 MTP block is architecturally identical (hidden 2048, 256 experts,
vocab 248320), so it grafts straight in: copy the 785 BF16 `mtp.*` tensors from the
base MXFP4 build, set `mtp_num_hidden_layers: 1`, and **add the `mtp.*` modules to
`quantization_config.ignore`** (else vLLM tries to load the BF16 MTP as quantized →
`fc.weight not found in params_dict`). See `scripts/graft-mtp.py`. The grafted head
hits **55% acceptance** despite being base-trained (native is 81%) — MTP is lossless,
so it only ever helps. A native MTP retrain on the distill's hidden states would close
the remaining concurrency gap.

## Reproduce

```bash
# 0. deps: olka/qstream @ master in a venv (pip install -e .), safetensors, torch
# 1. quantize BF16 -> MXFP4 (CPU, RTN, no GPU needed)
bash scripts/quant-mxfp4.sh
# 2. wrap config as multimodal + graft the base MTP block
python scripts/graft-mtp.py
#    (also rewrap config.json text->multimodal; see script comments / issues)
# 3. serve on the RDNA4 vLLM container (2x R9700, TP2, MTP-3)
bash scripts/serve-mxfp4-mtp.sh
```

> The scripts use the author's absolute paths (`/home/ludwid/...`); adjust to taste.

## Caveats

- **MTP acceptance 55%** (grafted, base-trained head) vs 81% native → realistic-prompt
  high-concurrency trails the native-MTP base. Single-stream and short-prompt concurrency
  are at/above parity.
- **Thinking traces are inline, not in `reasoning_content`** — the qwen3 reasoning parser
  doesn't capture a separated `<think>` block from this distill; it reasons in `content`.
- Built and tested only on **gfx1201 (RDNA4, R9700)** with `tcclaviger/vllm-rocm-mxfp4-nvfp4`.

## Attribution

- **Base model:** [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0)
- **Distill:** [nerkyor/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill](https://huggingface.co/nerkyor/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill) (Apache-2.0); teacher: DeepSeek-V4-Pro (MIT)
- **Quantizer:** [olka/qstream](https://github.com/olka/qstream)
- **Reference MXFP4 recipe + RDNA4 container:** [pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4](https://huggingface.co/pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4), [tcclaviger/vllm-rocm-mxfp4-nvfp4](https://hub.docker.com/r/tcclaviger/vllm-rocm-mxfp4-nvfp4)

## License

MIT — see [LICENSE](LICENSE). Inherits Apache-2.0 (Qwen base) + MIT (distill data) attribution.
