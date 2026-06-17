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

Both columns measured on the same bench (2× R9700, TP2, MTP-3):

| Metric | This build | Base (`pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`) |
|---|---|---|
| Single-stream, short prompt | **~107 tok/s** | ~101 |
| Single-stream, 6k prompt | ~82 tok/s | ~85 |
| Concurrency ceiling, short prompt | **~128** | ~128 |
| Aggregate @128 (short) | **~1875 tok/s** | ~1683 |
| MTP draft acceptance (MTP-3, measured) | ~56% (grafted) | ~64% (native) |
| Size | 69.3 GB BF16 → **20.5 GB** (29.6%) | — |

Effectively **at parity** — the distill edges short-prompt single-stream (107 vs 101) and
high-concurrency aggregate (1875 vs 1683 @128); 6k single-stream is a wash (82 vs 85). At 6k
prompts under high concurrency *both* models drop below the usable floor (a stack property, not
the distill). The distill's real win is **agentic: ~4× faster per turn-chain** at equal task
success — its DS-V4-Pro distillation makes it decisive.

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
hits **~56% acceptance** despite being base-trained — only ~8pp behind the base's own native
MTP (~64%, both measured at MTP-3). MTP is lossless, so it only ever helps; a native MTP
retrain on the distill's hidden states would close the rest.

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

> The scripts use `$HOME`-relative paths; set the model / output / venv dirs to taste.

## Optional: add image support back (vision graft)

The distill is text-only — nerkyor dropped the base's vision tower on export. But just like MTP,
the base Qwen3.6 vision tower **grafts straight back in** (`scripts/graft-vision.py`), giving a
text **+ reasoning + vision + MTP** model at **zero text-perf cost**:

1. Copy the base's 333 `model.visual.*` tensors (BF16, ~0.9 GB) into a new shard.
2. Add the `model.visual.*` modules to `quantization_config.ignore` (kept BF16, like MTP).
3. The config is already multimodal-shaped from the wrap step — just **drop `--language-model-only`**
   when serving, and use the base's `chat_template` + `preprocessor_config` (handles image tokens).

**Why it works:** the vision tower → merger projects to `out_hidden_size: 2048` = the LM's hidden
size, and the light text-only LoRA didn't disturb the LM's ability to read those embeddings.
Verified: fed a generated image (text + shapes), the model read the text, shapes, colors, and
positions correctly. Unlike MTP, vision is **not** lossless — a heavier-finetuned LM could fail to
ground the base embeddings — so **test it** (it works here; YMMV on other distills).

Validation of the vision build (vs the text-only one): SQL regression **136/137, 0 FAIL**; agent
eval **27/27, 0 FAIL**; single-stream **108.9 tok/s**, ceiling **~128**, MTP **57%** — all identical
to text-only. Vision adds ~0.9 GB VRAM and **0 text-perf** (the tower only fires on image input).
Published as a separate model: `Capicua25x/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4-Vision`.

## Caveats

- **MTP acceptance ~56%** (grafted) vs ~64% native, both measured at MTP-3 — only ~8pp behind.
  Single-stream and short-prompt concurrency are at/above parity.
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
