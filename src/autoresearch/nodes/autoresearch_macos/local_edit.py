"""Track G-only constant edits preserving Python inline comments."""
from __future__ import annotations

import io
import tokenize
from pathlib import Path

from autoresearch.worker.local_worker import _AppliedEdit, _assignment_pattern, _values_loosely_equal


def apply_track_g_edit(path: Path, param: str, old: str, new: str) -> _AppliedEdit | str:
    source = path.read_text(encoding="utf-8")
    match = _assignment_pattern(param).search(source)
    if match is None:
        return f"assignment '{param} = ...' not found in {path.name}"
    raw = match.group(2)
    try:
        comment = next((t for t in tokenize.generate_tokens(io.StringIO(raw).readline)
                        if t.type == tokenize.COMMENT), None)
    except (tokenize.TokenError, IndentationError) as exc:
        return f"invalid assignment for '{param}': {exc}"
    cut = comment.start[1] if comment else len(raw)
    value = raw[:cut].rstrip()
    if not _values_loosely_equal(value, old):
        return f"'{param}' currently has value {value!r}, expected {old!r}; skipping edit"
    replacement = new + raw[len(value):]
    patched = source[:match.start(2)] + replacement + source[match.end(2):]
    try:
        compile(patched, str(path), "exec")
    except (SyntaxError, ValueError) as exc:
        return f"invalid edit for '{param}': {exc}"
    path.write_text(patched, encoding="utf-8")
    return _AppliedEdit(path, param, old, new, source, patched)
