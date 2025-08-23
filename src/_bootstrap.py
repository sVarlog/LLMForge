from __future__ import annotations
import os, sys
from pathlib import Path

def _append_once(p: Path):
	if p.exists():
		s = str(p.resolve())
		if s not in sys.path:
			sys.path.insert(0, s)

# Priority 1: explicit WORKSPACE (set in Docker already)
ws = os.getenv("WORKSPACE")
if ws:
	_append_once(Path(ws))

# Priority 2: detect repo root by markers (pyproject.toml, .git, or a custom marker)
else:
	cur = Path(__file__).resolve()
	for parent in [cur] + list(cur.parents):
		if (parent / "pyproject.toml").exists() or (parent / ".git").exists() or (parent / ".project-root").exists():
			_append_once(parent)
			break