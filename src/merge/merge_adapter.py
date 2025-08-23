import json
import shutil
import re
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import _bootstrap  # normalizes sys.path so `import config` works everywhere
from config.config import MODEL_FAMILY, MODEL_NAME

# ───────────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────────

# BASE_MODEL_NAME = "{MODEL_FAMILY}/{MODEL_NAME}"
BASE_MODEL_NAME = f"{MODEL_FAMILY}/{MODEL_NAME}"
ADAPTER_ROOT = Path("output") / BASE_MODEL_NAME              # where train.py wrote runs
OUTPUT_ROOT = Path("merged-models") / f"{MODEL_FAMILY}"      # where to save merged model

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(msg, flush=True)

def _latest_subdir(root: Path) -> Path:
    subs = [p for p in root.iterdir() if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No subfolders under: {root.resolve()}")
    subs.sort(key=lambda p: p.stat().st_mtime)
    return subs[-1]

def find_last_training_run(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"No adapter runs found under: {root.resolve()}")
    return _latest_subdir(root)

def find_last_checkpoint(run_dir: Path) -> Path:
    """
    Return the latest checkpoint directory (numeric step).
    Prefers '-sanitized' variant if both exist for the same step.
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"No run dir: {run_dir}")

    def step_num(p: Path) -> int:
        m = re.match(r"^checkpoint-(\d+)", p.name)
        return int(m.group(1)) if m else -1

    cands = [p for p in run_dir.iterdir()
             if p.is_dir() and p.name.startswith("checkpoint-") and step_num(p) >= 0]
    if not cands:
        raise FileNotFoundError(f"No checkpoint-* found under {run_dir.resolve()}")

    # pick highest step
    max_step = max(step_num(p) for p in cands)
    same_step = [p for p in cands if step_num(p) == max_step]

    # prefer the sanitized folder if present
    for p in same_step:
        if p.name.endswith("-sanitized"):
            return p
    # otherwise return any (the plain one)
    return same_step[0]

def prepare_output_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    run_id = len(list(OUTPUT_ROOT.glob("merging-*"))) + 1
    out = base / f"merging-{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    return out

# ───────────────────────────────────────────────────────────────────────────────
# Adapter config sanitation (drops unknown keys like "corda_config")
# ───────────────────────────────────────────────────────────────────────────────
def _allowed_lora_keys() -> set:
    # Try to derive allowed keys directly from the installed PEFT
    try:
        import inspect
        from peft import LoraConfig
        sig = inspect.signature(LoraConfig.__init__)
        # include common base keys stored alongside LoRA params
        base_keys = {"peft_type", "task_type", "base_model_name_or_path", "revision", "inference_mode"}
        return set(k for k in sig.parameters.keys() if k != "self") | base_keys
    except Exception:
        # Fallback list (safe superset of common keys)
        return {
            "peft_type", "task_type", "base_model_name_or_path", "revision", "inference_mode",
            "r", "lora_alpha", "lora_dropout", "target_modules", "modules_to_save", "bias",
            "layers_to_transform", "layers_pattern", "rank_pattern", "alpha_pattern",
            "fan_in_fan_out", "init_lora_weights", "use_rslora", "use_dora", "lora_dtype",
            "tensor_parallel_size", "megatron_config",
        }

def _sanitize_one(cfg: dict, allowed: set) -> dict:
    # Drop any unknown keys (e.g., "corda_config") to avoid PEFT init errors
    sanitized = {k: v for k, v in cfg.items() if k in allowed}
    # Preserve the required peft/task markers if present
    if "peft_type" not in sanitized and "peft_type" in cfg:
        sanitized["peft_type"] = cfg["peft_type"]
    if "task_type" not in sanitized and "task_type" in cfg:
        sanitized["task_type"] = cfg["task_type"]
    return sanitized

def sanitize_adapter_config(ckpt: Path) -> Path:
    """
    Read adapter_config.json and remove keys PEFT doesn't accept.
    Creates a sanitized copy of the checkpoint folder if changes were needed.
    Returns the path to the folder that should be passed to PeftModel.from_pretrained.
    """
    cfg_path = ckpt / "adapter_config.json"
    if not cfg_path.exists():
        # Older PEFT versions may use different filenames, but if it's absent
        # there's nothing we can sanitize—just return the original.
        return ckpt

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    allowed = _allowed_lora_keys()
    changed = False

    if "peft_config" in data and isinstance(data["peft_config"], dict):
        new_data = {"peft_config": {}}
        for name, cfg in data["peft_config"].items():
            san = _sanitize_one(cfg, allowed)
            if san != cfg:
                changed = True
            new_data["peft_config"][name] = san
    else:
        new_data = _sanitize_one(data, allowed)
        if new_data != data:
            changed = True

    if not changed:
        return ckpt

    # Make a sanitized copy of the checkpoint directory
    out = ckpt.parent / (ckpt.name + "-sanitized")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ckpt, out)
    with (out / "adapter_config.json").open("w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    log("🧼 Sanitized adapter_config.json (removed unsupported keys).")
    return out

# ───────────────────────────────────────────────────────────────────────────────
# Merge
# ───────────────────────────────────────────────────────────────────────────────
def main():
    log("🔎 Locating latest training artifacts…")
    run_dir = find_last_training_run(ADAPTER_ROOT)
    ckpt = find_last_checkpoint(run_dir)
    log(f"  • Run dir:       {run_dir}")
    log(f"  • Checkpoint dir:{ckpt}")

    # Load the tokenizer actually used for training (sits at run_dir)
    log("\n🔤 Loading training tokenizer (from run root)…")
    train_tok = AutoTokenizer.from_pretrained(
        run_dir,
        use_fast=True,
        trust_remote_code=True,
    )
    log(f"  • Training tokenizer size: {len(train_tok)}")

    # Load base model
    log("\n🧱 Loading base model…")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Sanitize adapter config (drop unknown keys like "corda_config") and merge
    log("\n🪄 Attaching PEFT adapter and merging…")
    ckpt_for_load = sanitize_adapter_config(ckpt)
    peft_model = PeftModel.from_pretrained(base_model, ckpt_for_load, is_trainable=False)
    merged = peft_model.merge_and_unload()  # returns a plain HF model

    # Resize token embeddings if tokenizer grew during training
    current_vocab = merged.get_input_embeddings().weight.shape[0]
    target_vocab = len(train_tok)
    if target_vocab != current_vocab:
        log(f"\n📏 Resizing token embeddings: {current_vocab} → {target_vocab}")
        merged.resize_token_embeddings(target_vocab)
        try:
            merged.tie_weights()
        except Exception:
            pass
    else:
        log("\n📏 No resize needed (vocab unchanged).")

    # Save merged model + tokenizer
    out_dir = prepare_output_dir(OUTPUT_ROOT)
    log(f"\n💾 Saving merged model to: {out_dir}")
    merged.save_pretrained(out_dir, safe_serialization=True)
    log("💾 Saving tokenizer used in training…")
    train_tok.save_pretrained(out_dir)

    # Copy any extra tokenizer/training artifacts that help inference parity
    extras = (
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",       # for BPE tokenizers
        "merges.txt",       # for BPE tokenizers
        "tokenizer.json",   # for fast tokenizers
        "chat_template.jinja",  # if you used a custom template in train.py
    )
    log("\n🗂️ Copying auxiliary artifacts (if present):")
    copied_any = False
    for name in extras:
        src = run_dir / name
        if src.exists():
            shutil.copy(src, out_dir / name)
            log(f"  • {name}")
            copied_any = True
    if not copied_any:
        log("  (none found)")

    log("\n✅ Done! Merged model + tokenizer are ready.")
    log(f"   Path: {out_dir.resolve()}")

if __name__ == "__main__":
    main()