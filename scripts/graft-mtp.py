#!/usr/bin/env python
"""Graft the BF16 MTP draft block from pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4
into the DSV4Pro-Thinking-Distill-MXFP4, which shipped with mtp_num_hidden_layers=0
(nerkyor dropped it on export). Dims match exactly (hidden 2048, 256 experts,
vocab 248320), so the base 1-layer MTP head drops straight in.

  - copies all `mtp.*` tensors verbatim (they're BF16 in pahajoki; mtp is never
    quantized) into a new shard model-mtp.safetensors
  - patches model.safetensors.index.json weight_map + total_size
  - bumps config text_config.mtp_num_hidden_layers 0 -> 1

MTP is lossless (drafts verified against the main model), so worst case it just
doesn't accelerate. Acceptance may be below pahajoki's native 81% because the head
was trained on the BASE LM's hidden states, not the LoRA-shifted distill -> run MTP-1.
"""
import json, os, glob
from safetensors import safe_open
from safetensors.torch import save_file

PAH = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--pahajokiconsulting--Qwen3.6-35B-A3B-MXFP4/snapshots/*/"))[0]
OUT = os.path.expanduser("~/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4")

pah_idx = json.load(open(os.path.join(PAH, "model.safetensors.index.json")))
mtp_map = {k: v for k, v in pah_idx["weight_map"].items() if k.startswith("mtp.")}
print(f"mtp tensors to graft: {len(mtp_map)}")

by_shard = {}
for name, shard in mtp_map.items():
    by_shard.setdefault(shard, []).append(name)

tensors, total = {}, 0
for shard, names in by_shard.items():
    with safe_open(os.path.join(PAH, shard), framework="pt") as f:
        for n in names:
            t = f.get_tensor(n).contiguous()
            tensors[n] = t
            total += t.numel() * t.element_size()
print(f"loaded {len(tensors)} tensors, {total/1e9:.2f} GB  (dtype sample: {next(iter(tensors.values())).dtype})")

out_shard = "model-mtp.safetensors"
save_file(tensors, os.path.join(OUT, out_shard), metadata={"format": "pt"})
print(f"wrote {out_shard}")

dist_path = os.path.join(OUT, "model.safetensors.index.json")
dist_idx = json.load(open(dist_path))
for n in tensors:
    dist_idx["weight_map"][n] = out_shard
dist_idx["metadata"]["total_size"] = dist_idx["metadata"].get("total_size", 0) + total
json.dump(dist_idx, open(dist_path, "w"), indent=2)
print("index patched")

cfg_path = os.path.join(OUT, "config.json")
cfg = json.load(open(cfg_path))
tc = cfg.get("text_config", cfg)
old = tc.get("mtp_num_hidden_layers")
tc["mtp_num_hidden_layers"] = 1

# Add the mtp.* modules to quantization_config.ignore, else vLLM builds the MTP
# block as quantized (expects weight_packed) and the grafted BF16 tensors fail to
# load: "fc.weight not found in params_dict". Copy the reference build's mtp ignores.
pah_cfg = json.load(open(os.path.join(PAH, "config.json")))
mtp_ignores = [e for e in pah_cfg["quantization_config"]["ignore"] if "mtp" in e]
ig = cfg["quantization_config"]["ignore"]
cfg["quantization_config"]["ignore"] = sorted(set(ig) | set(mtp_ignores))
print(f"added {len(mtp_ignores)} mtp ignore entries (total {len(cfg['quantization_config']['ignore'])})")

json.dump(cfg, open(cfg_path, "w"), indent=2)
print(f"config mtp_num_hidden_layers {old} -> 1")
print("DONE")
