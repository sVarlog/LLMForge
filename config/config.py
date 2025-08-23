from pathlib import Path
import os

# Base model name
MODEL_FAMILY = "deepseek-ai"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# Keep MODEL_NAME as the model basename and build the full path from MODEL_FAMILY
# MODEL_NAME = "DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "deepseek-ai/DeepSeek-LLM-7B-Base" 
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Base model path (include family + name).
# If MODEL_NAME already contains an owner/family prefix (contains '/'), use it as-is
# to avoid duplicating the family twice (e.g. deepseek-ai/deepseek-ai/...)
if "/" in MODEL_NAME:
    BASE_MODEL_PATH = MODEL_NAME
else:
    BASE_MODEL_PATH = f"{MODEL_FAMILY}/{MODEL_NAME}"
# Dynamically find the latest checkpoint. NOTE: do NOT run filesystem side-effects at import time.
output_dir = Path(f"output/{BASE_MODEL_PATH}")

# ADAPTER_PATH will be resolved lazily by calling `resolve_adapter_checkpoint()` below.
ADAPTER_PATH = None

def resolve_adapter_checkpoint() -> Path | None:
    """Read-only lookup of latest training/checkpoint under output/<family>/<model>.

    This function never creates directories or files. It only inspects the filesystem
    and returns the latest checkpoint Path or None if nothing is present.
    """
    global ADAPTER_PATH
    out = output_dir
    if not out.exists():
        return None

    training_dirs = [step for step in out.iterdir() if step.is_dir() and step.name.startswith("training-")]
    if not training_dirs:
        return None

    try:
        latest_training_dir = max(training_dirs, key=lambda path: int(path.name.split("-")[1]))
    except Exception:
        latest_training_dir = training_dirs[-1]

    checkpoints = [step for step in latest_training_dir.iterdir() if step.is_dir() and step.name.startswith("checkpoint-")]
    if not checkpoints:
        return None

    try:
        latest_checkpoint = max(checkpoints, key=lambda path: int(path.name.split("-")[1]))
    except Exception:
        latest_checkpoint = checkpoints[-1]

    ADAPTER_PATH = latest_checkpoint
    return ADAPTER_PATH

# Merged model path
MERGED_MODEL_PATH = Path(f"merged-models/{MODEL_FAMILY}")

# Allowed keys for adapter configuration cleaning
ALLOWED_KEYS = {
    "peft_type", "base_model_name_or_path", "inference_mode", "r",
    "lora_alpha", "lora_dropout", "bias", "target_modules", "task_type",
    "modules_to_save", "rank_pattern", "alpha_pattern", "fan_in_fan_out",
    "init_lora_weights", "layers_to_transform", "layers_pattern",
    "auto_mapping", "revision", "use_dora", "use_rslora"
}
