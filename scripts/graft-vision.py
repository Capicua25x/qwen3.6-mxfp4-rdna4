#!/usr/bin/env python
"""EXPERIMENT: graft the base Qwen3.6 vision tower into the DSV4Pro-Thinking distill MXFP4,
to test whether the LoRA-shifted LM can still interpret base vision embeddings.

Non-destructive: builds a separate variant dir with HARDLINKED LM shards (no 21GB copy),
a new model-visual.safetensors from pahajoki's base, and merged config/index. The published
distill is untouched. Multimodal plumbing (chat_template, preprocessor) is taken from pahajoki's
base so the only variable under test is "does the distill LM read the vision tower's output".
"""
import json, os, glob, shutil
from safetensors import safe_open
from safetensors.torch import save_file

PAH = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--pahajokiconsulting--Qwen3.6-35B-A3B-MXFP4/snapshots/*/"))[0]
DIS = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Capicua25x--Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4/snapshots/*/"))[0]
VAR = os.path.expanduser("~/distill-vision-test")
os.makedirs(VAR, exist_ok=True)

# 1. extract the base vision tower (BF16) -> one shard
pah_idx = json.load(open(PAH + "/model.safetensors.index.json"))
vmap = {k: v for k, v in pah_idx["weight_map"].items() if "visual" in k}
by_shard = {}
for n, s in vmap.items():
    by_shard.setdefault(s, []).append(n)
tensors, total = {}, 0
for s, names in by_shard.items():
    with safe_open(os.path.join(PAH, s), framework="pt") as f:
        for n in names:
            t = f.get_tensor(n).contiguous(); tensors[n] = t; total += t.numel() * t.element_size()
save_file(tensors, os.path.join(VAR, "model-visual.safetensors"), metadata={"format": "pt"})
print(f"visual: {len(tensors)} tensors, {total/1e9:.2f} GB (dtype {next(iter(tensors.values())).dtype})")

# 2. hardlink the distill LM shards (no copy — same filesystem)
for f in os.listdir(DIS):
    if f.endswith(".safetensors"):
        dst = os.path.join(VAR, f)
        if not os.path.exists(dst):
            os.link(os.path.join(DIS, f), dst)

# 3. multimodal plumbing from pahajoki (chat template, image processor, tokenizer, gen config)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json"):
    src = os.path.join(PAH, f)
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(VAR, f))

# 4. config = distill's (already multimodal arch + vision_config) + visual ignore entries
cfg = json.load(open(DIS + "/config.json"))
pah_cfg = json.load(open(PAH + "/config.json"))
vis_ign = [e for e in pah_cfg["quantization_config"]["ignore"] if "visual" in e]
cfg["quantization_config"]["ignore"] = sorted(set(cfg["quantization_config"]["ignore"]) | set(vis_ign))
json.dump(cfg, open(VAR + "/config.json", "w"), indent=2)
print(f"added {len(vis_ign)} visual ignore entries")

# 5. index = distill's + visual tensor map
idx = json.load(open(DIS + "/model.safetensors.index.json"))
for n in tensors:
    idx["weight_map"][n] = "model-visual.safetensors"
idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + total
json.dump(idx, open(VAR + "/model.safetensors.index.json", "w"), indent=2)
print(f"variant built at {VAR}")
print(f"arch={cfg['architectures']}  has vision_config={'vision_config' in cfg}")
