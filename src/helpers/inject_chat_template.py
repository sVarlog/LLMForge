import json, sys
from pathlib import Path

model_dir = Path(sys.argv[1])  # e.g. merged-models/deepseek-ai/merging-1
tk_cfg = model_dir / "tokenizer_config.json"
tmpl_fp = model_dir / "chat_template.jinja"

if not tk_cfg.exists():
    raise FileNotFoundError(f"Missing {tk_cfg}")
if not tmpl_fp.exists():
    raise FileNotFoundError(f"Missing {tmpl_fp}")

cfg = json.loads(tk_cfg.read_text(encoding="utf-8"))
tmpl = tmpl_fp.read_text(encoding="utf-8").replace("\r\n", "\n")  # normalize newlines

# Inject / update the template
cfg["chat_template"] = tmpl
# (Optional) helpful defaults
cfg.setdefault("use_default_system_prompt", False)

tk_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Injected chat_template into {tk_cfg}")