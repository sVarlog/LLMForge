import re
import os
import json
import shutil
from time import time
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    logging, AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback, BitsAndBytesConfig,
)
from tokenizers import Tokenizer

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from config.config import MODEL_NAME

SYSTEM_PROMPT = "You are a structured assistant. Respond in exactly two parts using the format:\n<think>[Your reasoning]</think>\n<output>[Your answer]</output>"

DATA_PATH = "datasets/data.jsonl"
OUTPUT_BASE_DIR = Path(f"output/{MODEL_NAME}")
LORA_CONFIG_PATH = "config/lora_config.json"

def run_generation_and_print(model, tokenizer, messages, label=None):
    from transformers import StoppingCriteriaList, StoppingCriteria

    # 1) Define our criteria
    class MaxNewTokensCriteria(StoppingCriteria):
        def __init__(self, start_length, max_new_tokens):
            super().__init__()
            self.start_length = start_length
            self.max_new_tokens = max_new_tokens
        def __call__(self, input_ids, scores, **kwargs):
            return input_ids.shape[1] - self.start_length >= self.max_new_tokens

    class StopSequenceCriteria(StoppingCriteria):
        def __init__(self, stop_sequences_ids):
            super().__init__()
            # ensure list of lists
            self.stop_sequences_ids = [
                seq if isinstance(seq, (list, tuple)) else [seq]
                for seq in stop_sequences_ids
            ]
        def __call__(self, input_ids, scores, **kwargs):
            last_tokens = input_ids[0].tolist()
            for seq_ids in self.stop_sequences_ids:
                if seq_ids and last_tokens[-len(seq_ids):] == seq_ids:
                    return True
            return False

    # 2) Build the prompt
    prompt_text, tokenized = format_and_tokenize(
        messages, tokenizer, return_tensors=True, add_generation_prompt=True
    )
    if not prompt_text.strip().endswith("<|im_start|><|assistant|>"):
        print("⚠️ Generation prompt is not correctly appended.")

    # 3) Move inputs to device & compute prompt_length
    inputs = {k: v.to(model.device) for k, v in tokenized.items()}
    prompt_length = inputs["input_ids"].shape[1]

    # 4) Assemble stoppers
    stop_id = tokenizer.convert_tokens_to_ids("</output>")
    stoppers = StoppingCriteriaList([
        MaxNewTokensCriteria(start_length=prompt_length, max_new_tokens=256),
        StopSequenceCriteria([stop_id]),
    ])

    # 5) Generate with explicit overrides
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=False,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.2,
            stopping_criteria=stoppers,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 6) Decode & print
    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    header = f"\n🧪 {label}:\n" if label else "\n🧪 Generation:\n"
    print(header + decoded + "\n")
    print("Is structured output:", is_structured_output(decoded))
    print("-" * 60)

def check_lora_modules(model, lora_config_path: str):
    """
    Load the LoRA config from disk, then verify which target_modules
    names actually appear in model.named_modules(), printing a summary.
    """
    # 1) Load your LoRA config
    with open(lora_config_path, "r") as f:
        lora_cfg = LoraConfig(**json.load(f))

    # 2) Gather all module names
    all_module_names = [name for name, _ in model.named_modules()]

    # 3) Check each target
    found, missing = [], []
    print("\n🔍 Checking LoRA target modules against the model…")
    for target in lora_cfg.target_modules:
        matches = [mn for mn in all_module_names if target in mn]
        if matches:
            found.append(target)
            snippet = matches[:3] + (["…"] if len(matches) > 3 else [])
            print(f"  ✔ `{target}` matched in: {snippet}")
        else:
            missing.append(target)
            print(f"  ❌ `{target}` NOT found in model modules!")
    print(f"\n✅ Modules to be LoRA‐tuned : {found}")
    if missing:
        print(f"⚠️ Warning: these targets were missing and will be skipped: {missing}")
    print("==============================================\n")

    # Return the loaded config so you can immediately use it
    return lora_cfg

def log(msg):
    print(f"\n🔧 {msg}\n{'=' * 60}")

def prepare_output_dir() -> Path:
    existing_dirs = [d for d in os.listdir(OUTPUT_BASE_DIR) if d.startswith("training-")]
    next_training_num = len(existing_dirs) + 1
    output_dir = OUTPUT_BASE_DIR / f"training-{next_training_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_and_prepare_tokenizer(output_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True
    )
    # register every tag you emit in your template
    special_tokens = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|pad|>",
        "additional_special_tokens": [
            "<|system|>", "<|user|>", "<|assistant|>",
            "<think>", "</think>", "<output>", "</output>"
        ],
    }
    tokenizer.add_special_tokens(special_tokens)

    return tokenizer

def tokenize_function(ex, tokenizer):
    # Build the assistant response string
    if ex.get("think"):
        response = f"<think>{ex['think']}</think><output>{ex['output']}</output>"
    else:
        response = f"<output>{ex['output']}</output>"

    # Messages with system prompt + user question + assistant response
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": ex["question"]},
        {"role": "assistant", "content": response}
    ]

    # Format + tokenize
    _, tokenized = format_and_tokenize(messages, tokenizer)

    # Create labels masking pads
    tokenized["labels"] = [
        tok_id if tok_id != tokenizer.pad_token_id else -100
        for tok_id in tokenized["input_ids"]
    ]
    return tokenized

def format_and_tokenize(messages, tokenizer, return_tensors=False, add_generation_prompt=False):
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )
    if return_tensors:
        tokenized = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=False)
    else:
        tokenized = tokenizer(
            formatted_text,
            padding="longest",         
            truncation=True,
            max_length=2048,              # pick appropriate max length
            return_tensors=None,          # ✅ Don't return tensor now
            add_special_tokens=False,
        )
    return formatted_text, tokenized

def load_and_tokenize_dataset(tokenizer):
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"

    # 1) Load the flattened JSONL into a HuggingFace dataset
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 2) Tokenize & build labels
    dataset = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer),
        remove_columns=["id", "topic", "question", "think", "output"]
    )

    # 3) Debug prints if you like
    print("\nChat template preview:\n")
    print("-" * 50)
    print(tokenizer.chat_template[:200])
    print("-" * 50)
    print("\nSample decoded inputs:\n")
    print("-" * 50)
    print(tokenizer.decode(dataset[0]["input_ids"]))
    print("-" * 50)

    return dataset

def load_model_and_prepare_for_qora(tokenizer, output_dir: Path):
    start = time()
    log("Loading model config and weights…")

    # 1) Base config + 4-bit quant setup
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
    )

    # 2) Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb,
    )

    # 2.1) Snapshot everything we'll need downstream
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Saving base model config and tokenizer to {output_dir}")
    config.save_pretrained(output_dir)

    # 3) Resize & patch
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # 4) Prepare for QLoRA
    log("Preparing model for QLoRA adapters…")
    model = prepare_model_for_kbit_training(model)

    # 5) Load your LoRA config, wrap with PEFT…
    assert os.path.exists(LORA_CONFIG_PATH), "Missing LoRA config"
    log(f"Checking LoRA config at {LORA_CONFIG_PATH}…")
    lora_cfg = check_lora_modules(model, LORA_CONFIG_PATH)
    log("Applying LoRA adapters…")
    model = get_peft_model(model, lora_cfg)

    # 6) Monkey-patch generation defaults
    model.generation_config.do_sample   = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p       = 1.0
    model.generation_config.top_k       = 0
    model.config.use_cache              = False

    end = time()
    log(f"✅ Model & LoRA ready in {end - start:.2f}s")
    return model


def save_chat_jinja2(tokenizer, output_dir: Path):
    template_src = Path("templates/chat_template.jinja")
    assert template_src.exists(), f"Template missing at: {template_src}"

    # 1) add the chat template to your tokenizer
    with open(template_src, "r", encoding="utf-8") as f:
        template_text = f.read()
    tokenizer.chat_template = template_text
    tokenizer.init_kwargs["chat_template"] = template_text

    # 2) save the HF tokenizer (this writes tokenizer.json and tokenizer_config.json)
    tokenizer.save_pretrained(output_dir)

    # 3) now open the *fast* tokenizer to dump vocab + merges
    fast_tok = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    bpe = fast_tok.model  # this is a tokenizers.models.BPE

    # Option A) Let `.save()` do it for you, into a dedicated subfolder:
    bpe_folder = output_dir / "bpe-tokenizer"
    bpe_folder.mkdir(exist_ok=True)
    bpe.save(str(bpe_folder))  # writes bpe-tokenizer/vocab.json & bpe-tokenizer/merges.txt

    # Move them up one level if you like:
    (bpe_folder / "vocab.json").rename(output_dir / "vocab.json")
    (bpe_folder / "merges.txt").rename(output_dir / "merges.txt")
    bpe_folder.rmdir()  # remove the now-empty subfolder

    # Option B) (more explicit) manually dump:
    # with open(output_dir/"vocab.json", "w", encoding="utf-8") as vf:
    #     json.dump(bpe.get_vocab(), vf, ensure_ascii=False)
    # with open(output_dir/"merges.txt", "w", encoding="utf-8") as mf:
    #     for a, b in bpe.get_merges():
    #         mf.write(f"{a} {b}\n")

    print(f"✅ Chat template + vocab/merges dumped to {output_dir}")


def is_structured_output(text: str) -> bool:
    # Extract only the assistant block (between <|im_start|>assistant and <|im_end|>)
    match = re.search(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", text, re.DOTALL)
    if not match:
        return False

    assistant_text = match.group(1)
    return all(tag in assistant_text for tag in ["<think>", "</think>", "<output>", "</output>"])

class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, interval):
        self.tokenizer = tokenizer
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval != 0:
            return
        
        log(f"🔬 Running evaluation at step {state.global_step}...")

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "What is the capital of France?"}
        ]
        run_generation_and_print(
            kwargs["model"],
            self.tokenizer,
            messages,
            label=f"Eval @ step {state.global_step}"
        )

def train_model(model, tokenizer, dataset, output_dir):
    log("Configuring training arguments...")

    log("Checking model parameters for meta device...")
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            raise RuntimeError(f"❌ Parameter {name} is still on meta device!")

    log("Disabling reentrant checkpointing...")
    torch.utils.checkpoint._use_reentrant = False

    log("Disabling use_cache for training...")
    model.config.use_cache = False

    log("Instantiating Trainer...")
    model.config.use_cache = False
    torch.utils.checkpoint._use_reentrant = False

    # 2) Build a small custom collator
    def causal_collator(features):
    # 1) Turn each feature’s token IDs back into the full text string
        texts = [tokenizer.decode(f["input_ids"], skip_special_tokens=False) for f in features]

        # 2) Batch‐tokenize & pad in one go (this avoids the fast‐tokenizer warning)
        batch = tokenizer(
            texts,
            padding="longest",
            pad_to_multiple_of=8,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # 3) Create labels from the padded input_ids
        labels = batch["input_ids"].clone()
        # 4) Mask out the pad positions so they don’t contribute to loss
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return batch

    log("Creating causal collator...")
    # 3) Configure Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        max_steps=20,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=10,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        disable_tqdm=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        # you can still force batched lengths to multiple of 8 here,
        # but tokenizer.pad(…, pad_to_multiple_of=8) handles it.
    )

    log("Creating Trainer instance...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=causal_collator,
        callbacks=[EvalCallback(tokenizer, interval=20)],
    )

    log("Trainer instance created successfully.")
    # 4) Train & save
    trainer.train()
    log("Training completed successfully.")


    model.save_pretrained(output_dir)
    log("Model saved successfully.")


def test_training():
    from peft import PeftModel

    BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    OUTPUT_DIR = "output/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/training-9"

    log("Loading base model and tokenizer for testing...")

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    chat_template_path = Path(OUTPUT_DIR) / "chat_template.jinja"
    assert chat_template_path.exists(), f"Template missing at: {chat_template_path}"
    tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    tokenizer.init_kwargs["chat_template"] = tokenizer.chat_template

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, OUTPUT_DIR, is_trainable=False)

    examples = [
        "What is the capital of France?",
        "Who wrote 'To Kill a Mockingbird'?",
        "Explain the theory of relativity in simple terms.",
        "What is the boiling point of water?",
        "How do airplanes fly?"
    ]

    for i, question in enumerate(examples, start=1):
        log(f"Processing example {i}: {question}")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question}
        ]
        run_generation_and_print(
            model,
            tokenizer,
            messages,
            label=f"Example {i}"
        )

def main():
    log("Preparing output directory")
    output_dir = prepare_output_dir()

    log("Loading tokenizer and adding special tags")
    tokenizer = load_and_prepare_tokenizer(output_dir)

    log("Saving chat template to tokenizer")
    save_chat_jinja2(tokenizer, output_dir)

    log("Loading and tokenizing dataset")
    dataset = load_and_tokenize_dataset(tokenizer)

    log("Loading model and applying LoRA")
    model = load_model_and_prepare_for_qora(tokenizer, output_dir)

    print("=== Final Chat Template ===")
    print(tokenizer.chat_template)
    print("===========================")

    log("Training model")
    train_model(model, tokenizer, dataset, output_dir)

    # log('Testing training with a small dataset')
    # test_training()

if __name__ == "__main__":
    main()