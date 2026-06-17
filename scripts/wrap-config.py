#!/usr/bin/env python
"""Rewrap a text-only Qwen3.6-35B-A3B checkpoint's config.json into the multimodal
Qwen3_5MoeConfig shape the RDNA4 vLLM container's loader requires.

The tcclaviger RDNA4 container only registers Qwen3_5MoeForConditionalGeneration
(multimodal). A text-only checkpoint (Qwen3_5MoeForCausalLM / qwen3_5_moe_text)
crashes at processor build ("Invalid HF config ... found Qwen3_5MoeTextConfig").

Fix: nest the flat text fields under text_config, set model_type=qwen3_5_moe and
architectures=[Qwen3_5MoeForConditionalGeneration], and graft a vision_config (+ the
vision/image token ids) from a known-good multimodal build. Then serve with
--language-model-only so the vision tower is a no-init stub (its weights are never
loaded — this checkpoint has none). quantization_config stays at top level unchanged.

Usage: python wrap-config.py <model_dir> <reference_multimodal_model_dir>
  e.g. python wrap-config.py ./MyDistill-MXFP4 ~/.cache/.../pahajokiconsulting--Qwen3.6-35B-A3B-MXFP4/snapshots/<sha>
"""
import json, os, sys, shutil

OUT, REF = sys.argv[1], sys.argv[2]
cfg_path = os.path.join(OUT, "config.json")
D = json.load(open(cfg_path))
if D.get("model_type") == "qwen3_5_moe":
    print("already multimodal-wrapped; nothing to do"); sys.exit(0)
P = json.load(open(os.path.join(REF, "config.json")))

shutil.copy(cfg_path, cfg_path + ".textonly.bak")
wrapped = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {k: v for k, v in D.items()
                    if k not in ("quantization_config", "architectures", "transformers_version")},
    "vision_config": P["vision_config"],
    "image_token_id": P["image_token_id"],
    "video_token_id": P["video_token_id"],
    "vision_start_token_id": P["vision_start_token_id"],
    "vision_end_token_id": P["vision_end_token_id"],
    "tie_word_embeddings": D.get("tie_word_embeddings", False),
    "quantization_config": D.get("quantization_config"),
    "transformers_version": D.get("transformers_version"),
}
json.dump(wrapped, open(cfg_path, "w"), indent=2)
print(f"wrapped {cfg_path} -> multimodal (text_config + vision_config); backup at .textonly.bak")
