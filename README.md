# 🧠 LLM Fine-Tuning, Merging & GGUF Conversion

This repo shows how to:

1. **Fine-tune** a base LLM (e.g. Qwen/DeepSeek) with QLoRA
2. **Merge** the resulting LoRA adapter into the base model
3. **Convert** the merged model into GGUF (for `llama.cpp` / local inference)

---

## 📂 Project Layout

```
.
├── config/
│   ├── config.py            # Central paths & constants
│   └── lora_config.json     # LoRA hyperparameters
├── datasets_new/
│   ├── ai/
│   │   └── dataset.jsonl
│   ├── build_train_jsonl.py # Build/train dataset script (new scalable structure)
│   └── train_data.jsonl     # Combined training data (new)
├── merged-models/
│   └── deepseek-merged/     # Merged model outputs
├── output/
│   └── deepseek-ai/         # QLoRA training runs
└── src/
    ├── train.py             # QLoRA fine-tuning & generation helpers
    ├── merge_adapter.py     # Merge adapter → base model
    └── convert_to_gguf.sh   # GGUF conversion wrapper (wrapper lives in src/)
└── tools/
    └── llama/               # `transformers-to-gguf.py` & helpers
```

---

## 📖 Datasets

This project uses domain-specific datasets under `datasets_new/*` following a structured, extensible layout.

The full topic hierarchy, tags, and example questions are maintained in `datasets_new/structure.enriched.json` and summarized in `datasets_new/README.md` (see that file for contribution instructions, required file names, and structure details).

You can add your own datasets by following the contribution guidelines in `datasets_new/README.md` and placing new topic folders under `datasets_new/`.

## 🚀 Quickstart

### 1. Build your dataset

```powershell
python datasets_new/build_train_jsonl.py
```

This pulls in every `dataset.jsonl` (or topic files) under `datasets_new/*` and writes `datasets_new/train_data.jsonl`.

### 2. Train with QLoRA

```bash
python src/train.py
```

Outputs checkpoints under `output/deepseek-ai/TRAINING-N/checkpoint-M/`.  
Special/chat tokens, `tokenizer.json`, `vocab.json`, `merges.txt`, and your `chat_template.jinja` are saved there.

Notes on resuming training

-   To continue the last training run instead of starting a new one, set `TRAINING_NEW = False` in `config/training_config.py`.
-   The resume logic prefers an epoch-based continuation: the trainer reads the epoch recorded in the checkpoint's `trainer_state.json` and will extend training by `TRAINING_EXTRA_EPOCHS` (see `TRAINING_EPOCHS` and `TRAINING_EXTRA_EPOCHS` in `config/training_config.py`). This avoids issues with absolute `max_steps` when resuming from checkpoints.
-   The codebase also includes small helpers under `src/helpers/` (for example `build_messages.py` and `loggers.py`) to keep prompt construction and logging consistent when resuming and running generations.

Generation stopping

-   The training/generation utilities include a decoding-based stopper that looks for output delimiters like `</output>` (or the model's end token) in decoded text rather than relying solely on exact token-id sequences. This is more robust across tokenizers and prevents the model from emitting unwanted extra tokens after the intended end marker.

### 3. Merge LoRA into the base

```bash
python src/merge_adapter.py
```

-   Picks the **last** `training-*` / `checkpoint-*`
-   Reads the adapter’s added embedding rows (via `adapter_model.safetensors`)
-   Resizes the HF base model to match
-   Merges & unloads LoRA weights
-   Saves under `merged-models/deepseek-merged/merging-K/`
-   Copies across your **full** trained-tokenizer artifacts:
    -   `tokenizer.json`
    -   `vocab.json`
    -   `merges.txt`
    -   `special_tokens_map.json`
    -   `chat_template.jinja`

### 4. Convert to GGUF

```bash
bash scripts/convert_to_gguf.sh --outtype q8_0
```

-   Locates the latest `merged-models/.../merging-K/`
-   Runs `transformers-to-gguf.py` → emits `*.gguf` in `merging-K/gguf-output/`

---

## 📝 Why copy _all_ tokenizer files?

When you added custom special/chat tokens and a Jinja template:

-   **`tokenizer.json`** holds your merges + special tokens + chat_template
-   **`vocab.json`** + **`merges.txt`** define your BPE vocabulary
-   **`special_tokens_map.json`** maps names → IDs
-   **`chat_template.jinja`** is your prompt-format template

By shipping them alongside the merged model, you preserve _exactly_ the same tokenization and chat layout your fine-tune used.

---

## 🛠 Fine-Tuning Tips

-   Use small batches (2–4) with gradient accumulation 16–32
-   Train for 3–5 epochs on ~2–3K samples to start
-   Monitor loss & generations via the built-in eval callback

---

## 🎉 Results

-   Adapter merging “just worked” once we resized embeddings and carried over the custom tokenizer.
-   Downstream GGUF conversion now sees the proper `tokenizer.model` alongside JSON/BPE files.

---

<!-- Third-Party Code -->

## 🛠️ Third-Party Code

We include parts of the [llama.cpp](https://github.com/ggml-org/llama.cpp) project under its MIT license:

```bash
Copyright (c) 2023-2024 The ggml authors
Copyright (c) 2023 Georgi Gerganov
```

### Those files are included verbatim from llama.cpp and are subject to the same MIT terms:

-   `tools/llama/convert_hf_to_gguf.py`
-   `tools/llama/convert_hf_to_gguf_update.py`
-   `tools/llama/convert_llama_ggml_to_gguf.py`
-   `tools/llama/convert_lora_to_gguf.py`
-   `tools/llama/gguf-py/gguf/*`
