import re
import os
import json
import sys
import warnings
import logging as pylog  # stdlib logging
from time import time
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    logging,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
    StoppingCriteriaList, 
    StoppingCriteria
)
from tokenizers import Tokenizer

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
import _bootstrap  # normalizes sys.path so `import config` works everywhere
from config.config import MODEL_NAME

from helpers.persist_chat_template import persist_chat_template
from src.helpers.build_messages import build_messages
from src.helpers.loggers import log, debug
from config.training_config import (
    SYSTEM_PROMPT,
    DATA_PATH,
    OUTPUT_BASE_DIR,
    LORA_CONFIG_PATH,
    ANCHOR_INTO_OUTPUT,
    SUPERVISE_OUTPUT_ONLY,
    FINAL_LOG_FH,
    _ORIG_STDOUT,
    _ORIG_STDERR,
    ASSISTANT_OPEN_WITH_NL,
    ASSISTANT_OPEN_NO_NL,
    TRAINING_NEW,
    TRAINING_EPOCHS,
    TRAINING_EXTRA_EPOCHS,
    EVAL_QUESTIONS
)

MAX_LEN=2048

# --- Difficulty weighting (training + eval) ---
DIFFICULTY_TO_LOSS_WEIGHT = {1: 0.90, 2: 1.00, 3: 1.15, 4: 1.35, 5: 1.60}
DIFFICULTY_TO_EVAL_WEIGHT = {1: 1.00, 2: 1.15, 3: 1.30, 4: 1.50, 5: 1.75}

def _cast_diff(v, default=3):
    try:
        return int(v)
    except Exception:
        return default

def _meta_block(ex: dict) -> str:
    """
    Renders a compact metadata header so the model always sees topic/difficulty.
    """
    tags = ex.get("tags", [])
    if isinstance(tags, list):
        tags_str = ", ".join(map(str, tags))
    else:
        tags_str = str(tags) if tags is not None else ""
    return (
        "[META]\n"
        f"category: {ex.get('category','')}\n"
        f"subcategory: {ex.get('subcategory','')}\n"
        f"topic: {ex.get('topic','')}\n"
        f"content_type: {ex.get('content_type','')}\n"
        f"difficulty: {ex.get('difficulty','')}\n"
        f"tags: {tags_str}\n"
        "[/META]\n\n"
    )

def _extract_between(text: str, open_tag: str, close_tag: str) -> str:
    m = re.search(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), text, flags=re.DOTALL)
    return (m.group(1).strip() if m else "").strip()

class _TeeStream:
    """
    Tee writes to console AND finalLog.txt without breaking tqdm/Trainer formatting.
    """
    def __init__(self, stream, sink_fh_getter):
        self._stream = stream
        self._sink_getter = sink_fh_getter
    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            fh = self._sink_getter()
            if fh:
                fh.write(data)
        except Exception:
            pass
    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            fh = self._sink_getter()
            if fh:
                fh.flush()
        except Exception:
            pass

class SFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # pull labels + weights, do NOT feed weights to model.forward
        labels = inputs.pop("labels")
        loss_weight = inputs.pop("loss_weight", None)   # [batch]
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift so tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()      # [B, T-1, V]
        shift_labels = labels[..., 1:].contiguous()          # [B, T-1]

        # token-level loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                              shift_labels.view(-1))
        token_loss = token_loss.view(shift_labels.shape)     # [B, T-1]

        valid_mask = (shift_labels != -100).float()
        per_sample = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1e-6)  # [B]

        if loss_weight is not None:
            # normalize so big batches with big weights don't distort scale
            loss_weight = loss_weight.to(per_sample.device).float().view(-1)
            loss = (per_sample * loss_weight).sum() / loss_weight.sum().clamp_min(1e-6)
        else:
            loss = per_sample.mean()

        return (loss, outputs) if return_outputs else loss

class StopOnSubstring(StoppingCriteria):
    def __init__(self, tokenizer, substrings, start_len: int, window_tokens: int = 64):
        self.tokenizer = tokenizer
        self.substrings = substrings
        self.start_len = int(start_len)
        self.window = window_tokens

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids: tensor [batch, seq_len]
        seq = input_ids[0].tolist()
        seq_len = len(seq)
        # nothing generated yet
        if seq_len <= self.start_len:
            return False
        # take only tokens after prompt start_len (cap to window)
        tail_start = max(self.start_len, seq_len - self.window)
        tail = seq[tail_start: seq_len]
        text = self.tokenizer.decode(tail, skip_special_tokens=False)
        for sub in self.substrings:
            if sub in text:
                return True
        return False

def prepare_output_dir() -> Path:
    """Create and return a new training-N output directory under OUTPUT_BASE_DIR.

    This helper performs all filesystem creation here in train.py (not in config).
    It ensures the OUTPUT_BASE_DIR exists, picks the next training-N name, creates
    the folder and a base checkpoint-1 inside it, then returns the Path.
    """
    # Ensure base exists
    try:
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fallback if OUTPUT_BASE_DIR isn't a Path (older callers)
        os.makedirs(str(OUTPUT_BASE_DIR), exist_ok=True)

    # List existing training-* directories (use pathlib for robustness)
    existing_dirs = [d for d in OUTPUT_BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("training-")]
    # Determine next training number
    nums = []
    for d in existing_dirs:
        try:
            nums.append(int(d.name.split("-")[1]))
        except Exception:
            continue
    next_training_num = (max(nums) + 1) if nums else (len(existing_dirs) + 1)

    output_dir = OUTPUT_BASE_DIR / f"training-{next_training_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ensure there is a base checkpoint folder so other code can rely on it
    (output_dir / "checkpoint-1").mkdir(parents=True, exist_ok=True)

    return output_dir


def find_last_training_dir() -> Path | None:
    """Return the Path to the last training-N folder or None if none exists."""
    if not os.path.exists(OUTPUT_BASE_DIR):
        return None
    dirs = [d for d in os.listdir(OUTPUT_BASE_DIR) if d.startswith("training-") and os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))]
    if not dirs:
        return None
    nums = [int(d.split("-")[1]) for d in dirs if d.split("-")[1].isdigit()]
    if not nums:
        return None
    last = max(nums)
    return OUTPUT_BASE_DIR / f"training-{last}"


def find_latest_checkpoint(training_dir: Path) -> Path | None:
    """Return latest checkpoint folder inside training_dir or None."""
    if not training_dir or not training_dir.exists():
        return None
    chkp_dirs = [p for p in training_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not chkp_dirs:
        return None
    # pick highest numeric suffix
    def _num(p):
        try:
            return int(p.name.split("-")[1])
        except Exception:
            return -1
    chkp_dirs.sort(key=_num)
    return chkp_dirs[-1]


def load_and_prepare_tokenizer(output_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        add_bos_token=False,
        add_eos_token=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    return tokenizer


def find_token_sequence(token_ids, seq_ids):
    """Returns index where seq_ids starts in token_ids, or -1 if not found."""
    if not seq_ids:
        return -1
    for i in range(len(token_ids) - len(seq_ids) + 1):
        if token_ids[i : i + len(seq_ids)] == seq_ids:
            return i
    return -1


def tokenize_function(ex, tokenizer, canonical_assistant_ids):
    """
    Builds:
      - user message = [META] block + original question
      - assistant = <think>...</think><output>...</output>
      - returns input_ids / labels / attention_mask / loss_weight (scalar)
    """

    # Difficulty → loss weight
    diff_int = _cast_diff(ex.get("difficulty", 3))
    loss_weight = DIFFICULTY_TO_LOSS_WEIGHT.get(diff_int, 1.0)

    # Compose messages with a metadata header so model always sees topic/difficulty
    user_content = _meta_block(ex) + ex["question"]
    response = f"<think>{ex['think']}</think><output>{ex['output']}</output>"

    messages = build_messages(SYSTEM_PROMPT, user_content, response)

    # Tokenize via chat template
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        max_length=MAX_LEN,
        truncation=True,
    )

    im_end_marker = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    # Locate assistant-open marker
    start_pos = find_token_sequence(token_ids, canonical_assistant_ids)
    
    if start_pos == -1:
        cand_with_nl = tokenizer.encode(ASSISTANT_OPEN_WITH_NL, add_special_tokens=False)
        cand_no_nl   = tokenizer.encode(ASSISTANT_OPEN_NO_NL, add_special_tokens=False)
        start_pos = find_token_sequence(token_ids, cand_with_nl)
        used_marker = cand_with_nl
        
        if start_pos == -1:
            start_pos = find_token_sequence(token_ids, cand_no_nl)
            used_marker = cand_no_nl
        if start_pos == -1:
            log("❌ Could not find assistant marker in tokens (tokenize_function)")
            tail = token_ids[-120:] if len(token_ids) > 120 else token_ids
            log("tail ids:", tail)
            log("tail toks:", tokenizer.convert_ids_to_tokens(tail))

            raise AssertionError("❌ Could not find assistant marker in tokens")
    else:
        used_marker = canonical_assistant_ids

    start_idx = start_pos + len(used_marker)

    # end marker = stop before <|im_end|>
    end_idx = -1
    for i in range(start_idx, len(token_ids) - len(im_end_marker) + 1):
        if token_ids[i : i + len(im_end_marker)] == im_end_marker:
            end_idx = i
            break
    if end_idx == -1:
        end_idx = len(token_ids)

    # Labels
    labels = [-100] * len(token_ids)

    if SUPERVISE_OUTPUT_ONLY:
        out_open  = tokenizer.encode("<output>", add_special_tokens=False)
        out_close = tokenizer.encode("</output>", add_special_tokens=False)
        start_out = find_token_sequence(token_ids[start_idx:end_idx], out_open)
        end_out   = find_token_sequence(token_ids[start_idx:end_idx], out_close)
        if start_out != -1 and end_out != -1:
            o_s = start_idx + start_out
            o_e = start_idx + end_out + len(out_close)
            labels[o_s:o_e] = token_ids[o_s:o_e]
        else:
            labels[start_idx:end_idx] = token_ids[start_idx:end_idx]
            debug("Could not find explicit <output> tags — supervising whole assistant span")
    else:
        labels[start_idx:end_idx] = token_ids[start_idx:end_idx]

    attention_mask = [1] * len(token_ids)

    # Return loss_weight as a scalar (per sample)
    return {
        "input_ids": token_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "loss_weight": loss_weight,
    }


def format_and_tokenize(messages, tokenizer, return_tensors=False, add_generation_prompt=False, canonical_assistant_ids=None):
    """
    Produce formatted_text and tokenized object. If add_generation_prompt True
    we optionally anchor the prompt inside <output> to force generation there.
    canonical_assistant_ids is a list[int] used to verify template output.
    """
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )

    if add_generation_prompt:
        # Verify template produced the assistant open; best to check token-level
        if canonical_assistant_ids is not None:
            tmp = tokenizer(formatted_text, return_tensors=None, add_special_tokens=False)
            ids = tmp["input_ids"]
            if isinstance(ids, list):
                ids_list = ids
            elif isinstance(ids, (list, tuple)):
                ids_list = ids
            else:
                # when using fast tokenizer it may return different shape; normalize
                try:
                    ids_list = ids[0] if hasattr(ids, "__len__") else ids
                except Exception:
                    ids_list = ids
            pos = find_token_sequence(ids_list, canonical_assistant_ids)
            if pos == -1:
                log("⚠️ formatted_text does NOT contain canonical assistant marker")
                debug("formatted_text repr: " + repr(formatted_text[-200:]))
                debug("canonical_assistant_ids tokens: " + str(tokenizer.convert_ids_to_tokens(canonical_assistant_ids)))
            else:
                debug("formatted_text contains canonical assistant marker at token pos " + str(pos))
        # Optionally anchor inside output so generation starts at answer location:
        if ANCHOR_INTO_OUTPUT:
            # append safe anchor that matches training layout
            # formatted_text = formatted_text + "<think>"
            pass

    if return_tensors:
        tokenized = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=False)
    else:
        tokenized = tokenizer(
            formatted_text, padding="longest", truncation=True, max_length=MAX_LEN, return_tensors=None, add_special_tokens=False
        )

    return formatted_text, tokenized

def build_bad_words_ids(tokenizer):
    bad = [
        "<|im_start|>", "<|user|>", "<|system|>",
        "<|im_im|>", "[META]", "[/META]"
    ]
    out = []

    for s in bad:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: out.append(ids)
    
    return out

def run_generation_and_print(model, tokenizer, messages, canonical_assistant_ids=None, label="Eval", mode="auto"):
    """
    mode:
      - "auto": no anchor; model may output <think>... or just <output>...
      - "force_think": append "<think>" to encourage think->output
      - "output_only": append "<output>" to force no think section
    """
    formatted_text, inputs = format_and_tokenize(
        messages, tokenizer,
        return_tensors=True,
        add_generation_prompt=True,
        canonical_assistant_ids=canonical_assistant_ids
    )

    model_device = next(model.parameters()).device
    inputs = {k: (v.to(model_device) if isinstance(v, torch.Tensor) else v)
              for k, v in inputs.items()}

    bad_words_ids = build_bad_words_ids(tokenizer)
        
    prompt_len = inputs["input_ids"].shape[1]
    stop_subs = ["</output>", "<|im_end|>"]
    stopping_criteria = StoppingCriteriaList([StopOnSubstring(tokenizer, stop_subs, start_len=prompt_len)])

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=256,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=4,
            repetition_penalty=1.05,
            bad_words_ids=bad_words_ids,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(output[0][input_len:], skip_special_tokens=False)

    # Include the prompt too (tail to keep logs readable)
    def _tail(s, n=400):
        return s if len(s) <= n else s[-n:]
    prompt_tail = _tail(formatted_text, 800)
    header = f"\n🧪 {label}:\n" if label else "\n🧪 Generation:\n"
    out_str = (
        header +
        "📥 Prompt (tail):\n" + prompt_tail + "\n\n" +
        "📤 Output:\n" + decoded + "\n"
    )

    # Check structured output against the full returned string (prompt + decoded)
    try:
        structured_flag = is_structured_output(out_str)
    except Exception:
        structured_flag = is_structured_output(decoded)
    log(f"Is structured output: {structured_flag}")
    
    return out_str


def check_lora_modules(model, lora_config_path: str):
    with open(lora_config_path, "r") as f:
        lora_cfg = LoraConfig(**json.load(f))
    all_module_names = [name for name, _ in model.named_modules()]
    found, missing = [], []

    log("Checking LoRA target modules against the model…")

    for target in lora_cfg.target_modules:
        matches = [mn for mn in all_module_names if target in mn]
        if matches:
            found.append(target)
            snippet = matches[:3] + (["…"] if len(matches) > 3 else [])
            log(f"  ✔ `{target}` matched in: {snippet}")
        else:
            missing.append(target)
            log(f"  ❌ `{target}` NOT found in model modules!")
    log(f"✅ Modules to be LoRA‐tuned : {found}")

    if missing:
        log(f"⚠️ Warning: these targets were missing and will be skipped: {missing}")

    return lora_cfg


def load_model_and_prepare_for_qora(tokenizer, output_dir: Path):
    start = time()
    log("Loading model config and weights…")
    # reduce HF verbosity during heavy init to avoid config dumps
    logging.set_verbosity_warning()

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    # log(f"Saving base model config and tokenizer to {output_dir}")
    # config.save_pretrained(output_dir)

    model.config.pad_token_id = tokenizer.pad_token_id

    log("Preparing model for QLoRA adapters…")
    model = prepare_model_for_kbit_training(model)

    assert os.path.exists(LORA_CONFIG_PATH), "Missing LoRA config"
    log(f"Checking LoRA config at {LORA_CONFIG_PATH}…")
    lora_cfg = check_lora_modules(model, LORA_CONFIG_PATH)
    log("Applying LoRA adapters…")
    # if output_dir already contains a PEFT adapter, we can load it later via PeftModel.from_pretrained
    model = get_peft_model(model, lora_cfg)

    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0
    model.config.use_cache = False

    end = time()

    log(f"✅ Model & LoRA ready in {end - start:.2f}s")
    logging.set_verbosity_warning()

    return model


def is_structured_output(text: str) -> bool:
    m = re.search(r"<\|im_start\|><\|assistant\|>\s*(.*)", text, re.DOTALL)
    segment = m.group(1) if m else text
    has_think  = ("<think>" in segment and "</think>" in segment)
    has_output = ("<output>" in segment and "</output>" in segment)

    return has_output and has_think


class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, canonical_assistant_ids, output_dir, interval, raw_dataset):
        self.tokenizer = tokenizer
        self.canonical_assistant_ids = canonical_assistant_ids
        self.interval = interval
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        self.raw_dataset = raw_dataset

    def _pick_eval_sample(self, state):
        import random
        # bias sampling by difficulty weight so we see harder items more
        weights = []
        for ex in self.raw_dataset:
            d = _cast_diff(ex.get("difficulty", 3))
            weights.append(DIFFICULTY_TO_EVAL_WEIGHT.get(d, 1.0))
        idx = random.choices(range(len(self.raw_dataset)), weights=weights, k=1)[0]
        return self.raw_dataset[idx]

    def _score_output(self, pred_text: str, ref_text: str, diff: int):
        # grab <output>...</output>
        pred_out = _extract_between(pred_text, "<output>", "</output>")
        # quick lexical F1 (casefold + simple tokenization)
        tok = lambda s: re.findall(r"[a-z0-9]+", s.casefold())
        p = tok(pred_out)
        r = tok(ref_text or "")
        if not p and not r:
            f1 = 1.0
        elif not p or not r:
            f1 = 0.0
        else:
            ps, rs = set(p), set(r)
            inter = len(ps & rs)
            prec = inter / max(len(ps), 1)
            rec  = inter / max(len(rs), 1)
            f1 = (2*prec*rec)/(prec+rec+1e-9)

        struct = 1.0 if is_structured_output(pred_text) else 0.0
        base = 0.2*struct + 0.8*f1
        return base * DIFFICULTY_TO_EVAL_WEIGHT.get(diff, 1.0), {"f1": f1, "structured": struct}

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval != 0:
            return

        ex = self._pick_eval_sample(state)
        diff = _cast_diff(ex.get("difficulty", 3))
        user_content = _meta_block(ex) + ex["question"]

        messages = build_messages(SYSTEM_PROMPT, user_content)

        mode = "force_think" if state.global_step < 100 else "auto"
        output_str = run_generation_and_print(
            kwargs["model"], self.tokenizer, messages,
            canonical_assistant_ids=self.canonical_assistant_ids,
            label=f"Eval @ step {state.global_step} (diff={diff})",
            mode=mode
        )

        # score
        score, parts = self._score_output(output_str, ex.get("output",""), diff)

        log_dict = dict(state.log_history[-1]) if state.log_history else {}
        log_dict.update({
            "eval_difficulty": diff,
            "eval_score_weighted": round(float(score), 4),
            "eval_f1": round(float(parts["f1"]), 4),
            "eval_structured": float(parts["structured"]),
        })
        metrics_str = f"Metrics: {json.dumps(log_dict, indent=2)}\n\n"
        log_file = self.logs_dir / f"callback-{state.global_step}.txt"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(metrics_str)
            f.write(output_str)

        try:
            if FINAL_LOG_FH:
                FINAL_LOG_FH.write(metrics_str)
                FINAL_LOG_FH.write(output_str)
                FINAL_LOG_FH.flush()
        except Exception:
            pass


def train_model(model, tokenizer, dataset, output_dir, canonical_assistant_ids, train_dataset, resume_from_checkpoint: Path | None = None):
    log("Configuring training arguments...")
    pylog.getLogger("accelerate").setLevel(pylog.INFO)
    pylog.getLogger("peft").setLevel(pylog.INFO)

    for name, param in model.named_parameters():
        if param.device.type == "meta":
            raise RuntimeError(f"❌ Parameter {name} is still on meta device!")

    torch.utils.checkpoint._use_reentrant = False
    model.config.use_cache = False

    def pad_collator(features):
        # flatten & pad manually for seq fields; keep loss_weight as scalar
        def flatten1d(x):
            if hasattr(x, "flatten"):
                x = x.flatten()
            if hasattr(x, "tolist"):
                x = x.tolist()
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                x = [item for sublist in x for item in sublist]
            return x

        seq_keys = ["input_ids", "labels", "attention_mask"]
        max_len = max(len(flatten1d(f["input_ids"])) for f in features)

        batch = {k: [] for k in seq_keys}
        loss_weights = []

        for f in features:
            for k in seq_keys:
                pad_token = tokenizer.pad_token_id if k != "labels" else -100
                v = flatten1d(f[k])
                arr = v + [pad_token] * (max_len - len(v))
                batch[k].append(arr)
            lw = float(f.get("loss_weight", 1.0))
            loss_weights.append(lw)

        out = {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "loss_weight": torch.tensor(loss_weights, dtype=torch.float32),
        }
        return out

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=TRAINING_EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        group_by_length=True,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=pad_collator,
        callbacks=[EvalCallback(tokenizer, canonical_assistant_ids, output_dir, interval=20, raw_dataset=train_dataset)],
    )

    if resume_from_checkpoint:
        # Try to extend epochs when resuming rather than fiddling max_steps.
        try:
            ts = Path(resume_from_checkpoint) / "trainer_state.json"
            if ts.exists():
                st = json.load(open(ts, "r", encoding="utf-8"))
                current_epoch = float(st.get("epoch", 0.0) or 0.0)
            else:
                current_epoch = 0.0
        except Exception:
            current_epoch = 0.0

        # If resuming, extend target epochs by TRAINING_EXTRA_EPOCHS
        try:
            original_epochs = float(training_args.num_train_epochs or TRAINING_EPOCHS)
        except Exception:
            original_epochs = TRAINING_EPOCHS

        new_target_epochs = current_epoch + TRAINING_EXTRA_EPOCHS

        if new_target_epochs <= original_epochs:
            pass # No alternative calculation; assignment above is sufficient

        log(f"Resuming from checkpoint. current_epoch={current_epoch:.2f}, setting num_train_epochs -> {new_target_epochs:.2f}")
        training_args.num_train_epochs = new_target_epochs

        # re-create trainer with updated args so the Trainer picks up the new epoch target
        trainer.args = training_args
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
    else:
        trainer.train()
    model.save_pretrained(output_dir)

def init_training():
    log("Preparing output directory")
    resume_checkpoint = None
    # Try to read any existing adapter/checkpoint info from config (read-only)
    try:
        from config import config as cfg
        candidate = cfg.resolve_adapter_checkpoint()
        if candidate is not None and TRAINING_NEW is False:
            # If user asked to resume (TRAINING_NEW=False) prefer the config-found checkpoint
            resume_checkpoint = candidate
            # set output_dir to the parent training folder
            output_dir = candidate.parent
            log(f"Found existing adapter checkpoint via config: {candidate}")
        else:
            # Otherwise create or choose a new output dir here in train.py
            if not TRAINING_NEW:
                last = find_last_training_dir()
                if last is not None:
                    log(f"TRAINING_NEW is False — reusing last training dir: {last}")
                    output_dir = last
                    # find latest checkpoint inside that dir
                    ck = find_latest_checkpoint(output_dir)
                    if ck is not None:
                        resume_checkpoint = ck
                        log(f"Found latest checkpoint: {resume_checkpoint}")
                else:
                    log("TRAINING_NEW is False but no previous training dir found; creating new one")
                    output_dir = prepare_output_dir()
            else:
                output_dir = prepare_output_dir()
    except Exception:
        # On any failure just fall back to local logic
        if not TRAINING_NEW:
            last = find_last_training_dir()
            if last is not None:
                output_dir = last
                ck = find_latest_checkpoint(output_dir)
                if ck is not None:
                    resume_checkpoint = ck
            else:
                output_dir = prepare_output_dir()
        else:
            output_dir = prepare_output_dir()
    # Global log sink
    global FINAL_LOG_FH
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    FINAL_LOG_FH = open(logs_dir / "finalLog.txt", "a", encoding="utf-8")

    # === Tee stdout/stderr so tqdm + Trainer bars and prints land in finalLog.txt ===
    global _ORIG_STDOUT, _ORIG_STDERR
    _ORIG_STDOUT, _ORIG_STDERR = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(_ORIG_STDOUT, lambda: FINAL_LOG_FH)
    sys.stderr = _TeeStream(_ORIG_STDERR, lambda: FINAL_LOG_FH)

    # === Capture Python warnings into the sink ===
    warnings.simplefilter("default")  # show deprecations by default
    def _showwarning(message, category, filename, lineno, file=None, line=None):
        s = warnings.formatwarning(message, category, filename, lineno, line)
        try:
            if FINAL_LOG_FH:
                FINAL_LOG_FH.write(s)
                FINAL_LOG_FH.flush()
        except Exception:
            pass
        # also print to console (already teed)
        try:
            _ORIG_STDERR.write(s)
            _ORIG_STDERR.flush()
        except Exception:
            pass
    warnings.showwarning = _showwarning

    # === Keep Transformers logger quieter to avoid config spam ===
    pylog.basicConfig(level=pylog.WARNING)
    tf_logger = pylog.getLogger("transformers")
    tf_logger.setLevel(pylog.WARNING)

    # avoid duplicate handlers on reruns
    if not any(isinstance(h, pylog.StreamHandler) and getattr(h.stream, "name", "") == FINAL_LOG_FH.name
               for h in tf_logger.handlers if hasattr(h, "stream")):
        tf_logger.addHandler(pylog.StreamHandler(FINAL_LOG_FH))

    # Optional: leave HF internal verbosity at WARNING for clean logs
    logging.set_verbosity_warning()

    log("Loading tokenizer and adding special tags")
    tokenizer = load_and_prepare_tokenizer(output_dir)

    # Determine canonical assistant-open token ids by rendering a sample formatted prompt
    # and checking which variant exists in the tokenization.
    s_nl = tokenizer.encode(ASSISTANT_OPEN_WITH_NL, add_special_tokens=False)
    s_no = tokenizer.encode(ASSISTANT_OPEN_NO_NL, add_special_tokens=False)

    # Save chat template & tokenizer files (so canonical detection uses same template)
    log("Saving chat template to tokenizer")
    save_dir = output_dir
    
    log(f"🔧 Saving chat template + tokenizer to {save_dir}")

    persist_chat_template(tokenizer, save_dir)

    log(f"🔧 ✅ Chat template + tokenizer saved to {save_dir}\n" + "="*60)

    tmpl_path = Path(output_dir) / "chat_template.jinja"
    tokenizer.chat_template = tmpl_path.read_text(encoding="utf-8")

    tokenizer.init_kwargs["chat_template"] = tokenizer.chat_template
    
    # also dump BPE files for inspection
    fast_tok = Tokenizer.from_file(str(save_dir / "tokenizer.json"))
    bpe = fast_tok.model
    bpe_folder = save_dir / "bpe-tokenizer"
    bpe_folder.mkdir(exist_ok=True)
    bpe.save(str(bpe_folder))
    (bpe_folder / "vocab.json").rename(save_dir / "vocab.json")
    (bpe_folder / "merges.txt").rename(save_dir / "merges.txt")
    bpe_folder.rmdir()
    log(f"✅ Chat template + vocab/merges dumped to {save_dir}")

    # Build a small formatted prompt and detect which variant appears
    fmt, tok = format_and_tokenize(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "__DETECT__"}],
        tokenizer,
        return_tensors=True,
        add_generation_prompt=True,
        canonical_assistant_ids=None,
    )
    fmt_ids = tok["input_ids"][0].tolist()
    pos_no = find_token_sequence(fmt_ids, s_no)
    pos_nl = find_token_sequence(fmt_ids, s_nl)

    if pos_no != -1 and pos_nl == -1:
        canonical_assistant_ids = s_no
        debug("Canonical assistant marker: no-newline variant")
    elif pos_nl != -1 and pos_no == -1:
        canonical_assistant_ids = s_nl
        debug("Canonical assistant marker: newline variant")
    elif pos_nl != -1 and pos_no != -1:
        # prefer exact match with no newline if both present (rare)
        canonical_assistant_ids = s_no
        debug("Both variants in template; picking no-newline as canonical")
    else:
        # fallback: prefer s_nl if tokenizer tends to add newline (observed in your logs)
        canonical_assistant_ids = s_nl
        debug("No variant found in detection; defaulting to newline variant (best-effort)")

    log("Loading and tokenizing dataset")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    train_dataset = dataset

    # Map dataset with our tokenize_function wrapper that uses canonical ids
    def map_fn(ex):
        return tokenize_function(ex, tokenizer, canonical_assistant_ids)

    # Remove every original column except the model features we just produced
    remove_cols = [c for c in dataset.column_names if c not in ("input_ids","labels","attention_mask","loss_weight")]
    dataset = dataset.map(map_fn, remove_columns=remove_cols, batched=False)
    
    log(f"Dataset loaded with {len(dataset)} examples.")
    log(f"Sample tokenized example: {dataset[0]}")

    stop_ids = tokenizer.encode("</output>", add_special_tokens=False)
    log(f"stop ids: {stop_ids}, {tokenizer.convert_ids_to_tokens(stop_ids)}")

    log("Loading model and applying LoRA")
    model = load_model_and_prepare_for_qora(tokenizer, output_dir)

    log("Training model")
    train_model(model, tokenizer, dataset, output_dir, canonical_assistant_ids, train_dataset, resume_from_checkpoint=resume_checkpoint)

    try:
        # restore std streams first
        if _ORIG_STDOUT: sys.stdout = _ORIG_STDOUT
        if _ORIG_STDERR: sys.stderr = _ORIG_STDERR
    except Exception:
        pass
    try:
        def _detach_handlers_to(file_obj):
            for name in ("transformers", "peft", "accelerate"):
                lg = pylog.getLogger(name)
                for h in list(lg.handlers):
                    if getattr(h, "stream", None) is file_obj:
                        lg.removeHandler(h)
                        try:
                            h.flush()
                            h.close()
                        except Exception:
                            pass

        if FINAL_LOG_FH:
            FINAL_LOG_FH.flush()

            _detach_handlers_to(FINAL_LOG_FH)

            FINAL_LOG_FH.close()
    except Exception:
        pass

def start_training():
    log("=== Starting training run ===")
    init_training()

if __name__ == "__main__":
    start_training()