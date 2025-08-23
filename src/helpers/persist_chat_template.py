import json
from pathlib import Path

def persist_chat_template(tokenizer, output_dir: Path, template_path: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) resolve template text (passed path -> templates/chat_template.jinja -> chat_template.jinja -> tokenizer.chat_template)
    candidates = [template_path] if template_path else []
    candidates += [Path("templates/chat_template.jinja"), Path("chat_template.jinja")]
    template_text = None
    for c in candidates:
        if c and c.exists():
            template_text = c.read_text(encoding="utf-8")
            break
    if template_text is None and getattr(tokenizer, "chat_template", None):
        template_text = tokenizer.chat_template
    if template_text is None:
        raise FileNotFoundError(
            "No chat template found. Put one at templates/chat_template.jinja, "
            "pass template_path, or set tokenizer.chat_template."
        )

    # 2) write a plain file alongside the run (useful for tools and humans)
    (output_dir / "chat_template.jinja").write_text(template_text, encoding="utf-8")

    # 3) set it on the tokenizer (best-effort for newer HF)
    tokenizer.chat_template = template_text
    tokenizer.init_kwargs["chat_template"] = template_text
    tokenizer.save_pretrained(output_dir)

    # 4) **force** it into tokenizer_config.json for old HF versions
    cfg_path = output_dir / "tokenizer_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            cfg = {}
        cfg["chat_template"] = template_text
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")