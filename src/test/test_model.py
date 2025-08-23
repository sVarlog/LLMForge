from __future__ import annotations
import os, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config.config import MODEL_FAMILY, MODEL_NAME
from config.training_config import SYSTEM_PROMPT, EVAL_QUESTIONS, OUTPUT_BASE_DIR
from src.helpers.build_messages import build_messages
from src.train import run_generation_and_print
from src.helpers.loggers import log

MERGED_BASE = Path("merged-models") / MODEL_FAMILY

GRAMMAR_THINK_OUTPUT = r"""
root    ::= think output
think   ::= "<" "t" "h" "i" "n" "k" ">" text "</" "t" "h" "i" "n" "k" ">" "\n"?
output  ::= "<" "o" "u" "t" "p" "u" "t" ">" text "</" "o" "u" "t" "p" "u" "t" ">"
text    ::= { [\u0009\u000A\u000D\u0020-\u003B\u003D-\U0010FFFF] }
"""

from huggingface_hub import snapshot_download

def _hf_repo_id(family: str, name: str) -> str:
    name = name.strip()
    return name if "/" in name else f"{family}/{name}"

def _resolve_local_base_dir(repo_id: str, tok_dir: Path) -> Path | None:
    """
    Try to find a fully local copy of the base model in this order:
      1) BASE_MODEL_DIR env var
      2) A pointer file saved by training: <run_root>/base_model_dir.txt
      3) Hugging Face cache snapshot (local, offline)
    Returns a directory path or None if nothing local is found.
    """
    # 1) explicit env
    env_dir = os.getenv("BASE_MODEL_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    # 2) pointer dropped by training (optional)
    hint = tok_dir / "base_model_dir.txt"
    if hint.exists():
        p = Path(hint.read_text(encoding="utf-8").strip())
        if p.exists():
            return p

    # 3) cached snapshot (offline)
    try:
        snap = snapshot_download(repo_id, local_files_only=True)
        return Path(snap)
    except Exception:
        return None

# ───────────────────────────────────────────────────────────────────────────────
# Filesystem resolvers
# ───────────────────────────────────────────────────────────────────────────────

def _latest_training_run(root: Path) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"No OUTPUT_BASE_DIR: {root}")
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("training-")]
    if not runs: raise FileNotFoundError(f"No training-* under {root}")
    runs.sort(key=lambda p: p.stat().st_mtime)
    return runs[-1]

def _latest_checkpoint(run_dir: Path) -> Path:
    cps = [d for d in Path(run_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not cps: raise FileNotFoundError(f"No checkpoint-* under {run_dir}")
    # accept checkpoint-123 or checkpoint-123-sanitized
    def step(p: Path) -> int:
        m = re.match(r"^checkpoint-(\d+)", p.name)
        return int(m.group(1)) if m else -1
    max_step = max(step(p) for p in cps)
    top = [p for p in cps if step(p) == max_step]
    # prefer sanitized if present
    for p in top:
        if p.name.endswith("-sanitized"): return p
    return top[0]

def _latest_merging_dir(base: Path = MERGED_BASE) -> Path:
    base = Path(base)
    if not base.exists():
        raise FileNotFoundError(f"No merged-models base for family: {base}")
    cands = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("merging-")]
    if not cands: raise FileNotFoundError(f"No 'merging-*' under {base}")
    # sort by numeric suffix if present, then mtime
    def num(p: Path):
        m = re.match(r"^merging-(\d+)$", p.name)
        return int(m.group(1)) if m else -1
    cands.sort(key=lambda p: (num(p), p.stat().st_mtime))
    # choose the newest that looks like a valid HF dir (has config.json)
    for p in reversed(cands):
        if (p / "config.json").exists():
            return p
    raise FileNotFoundError(f"No valid HF merged dir (config.json) found under {base}")

def _latest_gguf_file(merged_dir: Path) -> Path:
    gdir = Path(merged_dir) / "gguf-output"
    files = sorted([p for p in gdir.glob("*.gguf")], key=lambda p: p.stat().st_mtime)
    if not files: raise FileNotFoundError(f"No *.gguf in {gdir}")
    return files[-1]

def _resolve_adapter_paths(model_dir: Path) -> tuple[Path, Path]:
    """
    Returns (adapter_weights_dir, tokenizer_dir).
    If model_dir is a checkpoint folder, tokenizer_dir is its parent run dir.
    If model_dir is the run root (training-*), picks latest checkpoint.
    """
    model_dir = Path(model_dir)
    if model_dir.name.startswith("checkpoint-"):
        tok_dir = model_dir.parent   # run root
        return model_dir, tok_dir
    if model_dir.name.startswith("training-"):
        ckpt = _latest_checkpoint(model_dir)
        return ckpt, model_dir
    # direct checkpoint folder or adapter dump
    return model_dir, (model_dir if (model_dir / "tokenizer.json").exists() else model_dir.parent)

# ───────────────────────────────────────────────────────────────────────────────
# Tokenizer + HF loaders
# ───────────────────────────────────────────────────────────────────────────────

def _load_tokenizer_from(dir_path: Path):
    log(f"🔧 Loading tokenizer from: {dir_path}")
    tok = AutoTokenizer.from_pretrained(dir_path.as_posix(), trust_remote_code=True)
    tpl = dir_path / "chat_template.jinja"
    if tpl.exists():
        log(f"  • Using chat template: {tpl}")
        tok.chat_template = tpl.read_text(encoding="utf-8")
        tok.init_kwargs["chat_template"] = tok.chat_template
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok

def _eval_loop_hf(model, tokenizer, mode: str):
    try: model.eval()
    except Exception as e: log(f"(eval ignored) {e}")
    samples = [s for s in os.getenv("TEST_SAMPLES","").split(";") if s.strip()] or EVAL_QUESTIONS
    for i, question in enumerate(samples, start=1):
        messages = build_messages(SYSTEM_PROMPT, question.strip())
        out_str = run_generation_and_print(
            model, tokenizer, messages,
            canonical_assistant_ids=None, label=f"Example {i}", mode=mode
        )
        print(out_str)

def _eval_hf_adapter(adapter_dir: Path, tok_dir: Path, mode: str):
    tokenizer = _load_tokenizer_from(tok_dir)

    repo_id = _hf_repo_id(MODEL_FAMILY, MODEL_NAME)
    base_dir = _resolve_local_base_dir(repo_id, tok_dir)

    if base_dir is None:
        raise RuntimeError(
            "Base model not available locally.\n"
            "To run test-training without Hugging Face network access, you must provide a local base:\n"
            "  • Set BASE_MODEL_DIR to a local copy of the base (preferred), or\n"
            "  • Ensure the base is present in your HF cache (mount ~/.cache/huggingface), or\n"
            "  • Save a pointer file '<run_root>/base_model_dir.txt' with the local path.\n"
            f"Expected repo id: {repo_id}\n"
            f"Adapter: {adapter_dir}\n"
        )

    log(f"🧱 Loading base (local): {base_dir} and attaching adapter {adapter_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        base_dir.as_posix(),
        trust_remote_code=True,
        local_files_only=True,   # <<< no network
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, adapter_dir.as_posix(), is_trainable=False)
    _eval_loop_hf(model, tokenizer, mode)

def _eval_hf_merged(merged_dir: Path, mode: str):
    tokenizer = _load_tokenizer_from(merged_dir)
    log(f"🧱 Loading merged HF model from: {merged_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir.as_posix(),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    _eval_loop_hf(model, tokenizer, mode)

def _eval_gguf(gguf_path: Path, mode: str):
    """
    Evaluate a GGUF by rendering chat with the SAME HF tokenizer+template used elsewhere.
    This removes template drift and fixes stop mismatch (<|im_end|> etc).
    """
    import re
    from llama_cpp import Llama

    # same shape check as HF path
    SHAPE_RE = re.compile(r"^\s*<think>.*?</think>\s*<output>.*?</output>\s*$", re.S)
    def shape_ok(txt: str) -> bool:
        return bool(SHAPE_RE.match(txt))

    # n_ctx: keep modest unless you re-convert with rope-scaling; 8192 is safe
    n_ctx = int(os.getenv("TEST_CTX", "8192"))
    n_gpu_layers = int(os.getenv("N_GPU_LAYERS", "0"))

    # derive merged dir from .../merging-N/gguf-output/file.gguf
    merged_dir = gguf_path.parent.parent
    tok = _load_tokenizer_from(merged_dir)  # your existing helper

    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    # build conservative stop list
    stop_list = []
    for s in ["<|im_end|>", "</s>", getattr(tok, "eos_token", None)]:
        if s and s not in stop_list:
            stop_list.append(s)
            if not s.endswith("\n"):
                stop_list.append(s + "\n")

    samples = [s for s in os.getenv("TEST_SAMPLES", "").split(";") if s.strip()] or EVAL_QUESTIONS

    for i, question in enumerate(samples, start=1):
        messages = build_messages(SYSTEM_PROMPT, question.strip())
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        out = llm.create_completion(
            prompt=prompt,
            temperature=0.0,
            max_tokens=256,
            stop=stop_list,
        )
        text = out["choices"][0]["text"].strip()

        print("\n🔧 Is structured output:", shape_ok(text))
        print("============================================================\n")
        print(f"🧪 Example {i}:")
        print("📥 Prompt (tail):")
        print(f"<|im_start|><|system|>\n{SYSTEM_PROMPT}\n<|im_end|>")
        print(f"<|im_start|><|user|>\n{question}\n<|im_end|>\n")
        print("<|im_start|><|assistant|>\n")
        print("📤 Output:\n")
        print(text)
        print("\n")

# ───────────────────────────────────────────────────────────────────────────────
# Public entry points
# ───────────────────────────────────────────────────────────────────────────────

def run_test_training(mode: str = "auto"):
    """Latest training-N / checkpoint-M (adapter)"""
    run_dir = _latest_training_run(Path(OUTPUT_BASE_DIR))
    ckpt_dir = _latest_checkpoint(run_dir)
    adapter_dir, tok_dir = _resolve_adapter_paths(ckpt_dir)
    _eval_hf_adapter(adapter_dir, tok_dir, mode)

def run_test_merging(mode: str = "auto"):
    """Latest merging-N (merged HF)"""
    merged_dir = _latest_merging_dir(MERGED_BASE)
    _eval_hf_merged(merged_dir, mode)

def run_test_gguf(mode: str = "auto"):
    """Latest merging-N/gguf-output/*.gguf"""
    merged_dir = _latest_merging_dir(MERGED_BASE)
    gguf = _latest_gguf_file(merged_dir)
    _eval_gguf(gguf, mode)