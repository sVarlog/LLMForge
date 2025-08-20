# training_config.py
"""
Centralized configuration for training constants and settings.
Import this in both train.py and helpers/loggers.py.
"""
from pathlib import Path
from config.config import MODEL_NAME

SYSTEM_PROMPT = (
    "You are a structured assistant. Respond in exactly two parts using the format:\n"
    "<think>[Your reasoning]</think>\n<output>[Your answer]</output>"
)

# DATA_PATH = "datasets/data.jsonl" # Old datasets
DATA_PATH = "datasets_new/train_data.jsonl"
OUTPUT_BASE_DIR = Path(f"output/{MODEL_NAME}")
LORA_CONFIG_PATH = "config/lora_config.json"

ASSISTANT_OPEN_WITH_NL = "<|im_start|><|assistant|>\n"
ASSISTANT_OPEN_NO_NL = "<|im_start|><|assistant|>"

# If True, anchor generation prompt inside <output> to force answer tokens
ANCHOR_INTO_OUTPUT = True

# If True, supervise ONLY the <output>...</output> span instead of entire assistant block.
SUPERVISE_OUTPUT_ONLY = False

# Debugging toggle
DEBUG = True
DEBUG_SAMPLE_LIMIT = 10
DEBUG_SAMPLE_RANDOM = False
DEBUG_SAMPLE_PROB = 0.05
_DEBUG_SEEN = 0
DEF_LOG_PREFIX = "🔧 "
DEF_DBG_PREFIX = "🐞 "
FINAL_LOG_FH = None
_ORIG_STDOUT = None
_ORIG_STDERR = None
TEE_ACTIVE = False  # set True after we install the tee streams

EVAL_QUESTIONS = [
  "2+2?",
  "Translate 'focus' to Polish.",
  "Is 7 > 5?",
  "Capital of France?",
]
