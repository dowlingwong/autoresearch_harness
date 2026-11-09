"""Code-level diff support for GPU nodes (G.2; R1-BX-D4; M10 step 1).

:func:`apply_diff` applies a unified diff inside a **per-run working copy**:

1. the diff is parsed strictly (no binary patches, renames, copies or mode
   changes; hunk line counts must match);
2. every file it touches is checked by the **existing scope validator**
   :func:`autoresearch.control_plane.permissions.validate_edit_scope`, so a
   hunk outside ``editable_paths`` or inside ``frozen_paths`` refuses the
   whole diff; paths that are absolute, contain ``..`` or resolve outside the
   working copy (symlinks) are refused before that;
3. all hunks are applied in memory with zero fuzz (a hunk whose context is
   not at its stated line is accepted only at a unique exact match elsewhere,
   and the offset is recorded); nothing is written unless every hunk applies;
4. **inert detection**: the diff is inert if no file changes after
   normalisation -- for ``.py`` files the AST (``ast.dump``) is compared, which
   ignores whitespace and comments; for other files, lines are compared with
   trailing whitespace, blank lines and ``#`` comment lines removed.  An inert
   diff is not written;
5. **duplicate detection** against an :class:`EvaluatedStates` registry: the
   normalised diff hash, and the hash of the resulting code state (the same
   normalisation over every editable file of the copy);
6. **provenance**: the diff text and its hash, the normalised diff hash, the
   files touched, the hunks with any offsets, and the incumbent and resulting
   code-state hashes (normalised and raw).

Nothing here is used by the T0 path (``manager_proposal/v1``).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from autoresearch.control_plane.permissions import validate_edit_scope
from autoresearch.nodes.spec import NodeSpec

DIFF_PROVENANCE_SCHEMA = "manager_proposal/v2-diff"
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_REFUSED_GIT_HEADERS = ("rename from", "rename to", "copy from", "copy to", "similarity index",
                        "dissimilarity index", "old mode", "new mode", "GIT binary patch", "Binary files")
_ALLOWED_GIT_HEADERS = ("diff --git ", "index ", "new file mode", "deleted file mode")
_HASH_COMMENT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".txt", ".conf", ""}
_SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
_SKIP_SUFFIXES = {".pyc", ".pyo"}


class DiffRefused(ValueError):
    """A diff that cannot be applied; ``reason`` is a stable machine-readable code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[tuple[str, str]] = field(default_factory=list)  # (tag, text) with tag in " -+"
    old_no_eol: bool = False
    new_no_eol: bool = False

    def old_block(self) -> list[str]:
        return [text for tag, text in self.lines if tag in " -"]

    def new_block(self) -> list[str]:
        return [text for tag, text in self.lines if tag in " +"]


@dataclass
class FilePatch:
    old_path: str | None  # None for a created file (/dev/null)
    new_path: str | None  # None for a deleted file
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path if self.new_path is not None else self.old_path  # type: ignore[return-value]


# --------------------------------------------------------------------------- parsing


def parse_unified_diff(diff_text: str) -> list[FilePatch]:
    """Parse a unified diff strictly.  Raises :class:`DiffRefused`."""
    if not isinstance(diff_text, str):
        raise DiffRefused("malformed_diff", "diff must be text")
    lines = diff_text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    patches: list[FilePatch] = []
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(_REFUSED_GIT_HEADERS):
            raise DiffRefused("unsupported_diff_feature", line.strip())
        if line.strip() == "" or line.startswith(_ALLOWED_GIT_HEADERS):
            i += 1
            continue
        if not line.startswith("--- "):
            raise DiffRefused("malformed_diff", f"unexpected line {i + 1}: {line[:80]!r}")
        if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
            raise DiffRefused("malformed_diff", f"'---' without '+++' at line {i + 1}")
        old_path = _header_path(line[4:], "a/")
        new_path = _header_path(lines[i + 1][4:], "b/")
        if old_path is None and new_path is None:
            raise DiffRefused("malformed_diff", "both sides are /dev/null")
        if old_path is not None and new_path is not None and old_path != new_path:
            raise DiffRefused("unsupported_diff_feature", f"rename {old_path} -> {new_path}")
        patch = FilePatch(old_path=old_path, new_path=new_path)
        if patch.path in seen:
            raise DiffRefused("malformed_diff", f"file {patch.path} appears twice")
        seen.add(patch.path)
        i += 2
        while i < len(lines) and lines[i].startswith("@@"):
            hunk, i = _parse_hunk(lines, i)
            patch.hunks.append(hunk)
        if not patch.hunks:
            raise DiffRefused("malformed_diff", f"no hunks for {patch.path}")
        patches.append(patch)
    return patches


def _header_path(raw: str, prefix: str) -> str | None:
    text = raw.split("\t", 1)[0].strip()
    if text == "/dev/null":
        return None
    if text.startswith('"') or "\\" in text:
        raise DiffRefused("unsafe_path", f"quoted or escaped path {text!r}")
    if text.startswith(prefix):
        text = text[len(prefix):]
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or text in {".", ""}:
        raise DiffRefused("unsafe_path", f"path {text!r} must be relative and inside the node")
    return str(pure)


def _parse_hunk(lines: list[str], i: int) -> tuple[Hunk, int]:
    match = _HUNK_RE.match(lines[i])
    if not match:
        raise DiffRefused("malformed_hunk", f"bad hunk header at line {i + 1}: {lines[i][:80]!r}")
    old_start, new_start = int(match.group(1)), int(match.group(3))
    old_count = int(match.group(2)) if match.group(2) is not None else 1
    new_count = int(match.group(4)) if match.group(4) is not None else 1
    hunk = Hunk(old_start, old_count, new_start, new_count)
    i += 1
    old_seen = new_seen = 0
    last_tag = ""
    while i < len(lines) and (old_seen < old_count or new_seen < new_count):
        line = lines[i]
        if line.startswith("\\"):
            _mark_no_eol(hunk, last_tag)
            i += 1
            continue
        tag, text = (line[0], line[1:]) if line else (" ", "")  # an empty line is empty context
        if tag not in " -+":
            raise DiffRefused("malformed_hunk", f"bad hunk line {i + 1}: {line[:80]!r}")
        if tag in " -":
            old_seen += 1
        if tag in " +":
            new_seen += 1
        hunk.lines.append((tag, text))
        last_tag = tag
        i += 1
    if old_seen != old_count or new_seen != new_count:
        raise DiffRefused("malformed_hunk", f"hunk at line {hunk.old_start}: counts do not match its header")
    if i < len(lines) and lines[i].startswith("\\"):
        _mark_no_eol(hunk, last_tag)
        i += 1
    return hunk, i


def _mark_no_eol(hunk: Hunk, last_tag: str) -> None:
    if last_tag == "-":
        hunk.old_no_eol = True
    elif last_tag == "+":
        hunk.new_no_eol = True
    elif last_tag == " ":
        hunk.old_no_eol = hunk.new_no_eol = True


# --------------------------------------------------------------------------- normalisation and hashing


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalise_line(line: str, suffix: str) -> str | None:
    stripped = line.rstrip().expandtabs(4)
    if not stripped.strip():
        return None
    if suffix in _HASH_COMMENT_SUFFIXES and stripped.lstrip().startswith("#"):
        return None
    return stripped


def normalised_text(text: str, suffix: str) -> str:
    """Whitespace- and comment-insensitive form of a non-Python file (or unparsable Python)."""
    out = [n for n in (_normalise_line(line, suffix) for line in text.split("\n")) if n is not None]
    return "\n".join(out)


def semantic_form(path: str, text: str | None) -> str:
    """The form compared for inertness and hashed for the code state."""
    if text is None:
        return "<absent>"
    suffix = PurePosixPath(path).suffix
    if suffix == ".py":
        try:
            return "AST:" + ast.dump(ast.parse(text), include_attributes=False)
        except (SyntaxError, ValueError):
            return "TEXT:" + normalised_text(text, suffix)
    return "TEXT:" + normalised_text(text, suffix)


def normalised_diff_sha256(patches: Iterable[FilePatch]) -> str:
    """Hash of the diff's removed and added lines per file, without headers, line numbers,
    context, blank lines, comment lines or trailing whitespace."""
    payload = []
    for patch in sorted(patches, key=lambda p: p.path):
        suffix = PurePosixPath(patch.path).suffix
        removed, added = [], []
        for hunk in patch.hunks:
            for tag, text in hunk.lines:
                norm = _normalise_line(text, suffix)
                if norm is None:
                    continue
                if tag == "-":
                    removed.append(norm)
                elif tag == "+":
                    added.append(norm)
        kind = "create" if patch.old_path is None else "delete" if patch.new_path is None else "modify"
        payload.append([patch.path, kind, removed, added])
    return _sha256_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def editable_files(work_dir: Path, node_spec: NodeSpec) -> list[str]:
    """Relative paths of all files in *work_dir* that the scope validator accepts as editable."""
    out = []
    root = Path(work_dir)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if PurePosixPath(name).suffix in _SKIP_SUFFIXES:
                continue
            rel = Path(dirpath, name).relative_to(root).as_posix()
            if validate_edit_scope([rel], node_spec).valid:
                out.append(rel)
    return sorted(out)


def code_state_hashes(work_dir: Path, node_spec: NodeSpec,
                      overrides: dict[str, str | None] | None = None) -> tuple[str, str]:
    """(normalised, raw) sha256 of the editable code state of *work_dir*.

    *overrides* maps relative paths to new contents (``None`` = deleted) so the
    resulting state can be hashed before anything is written.
    """
    overrides = overrides or {}
    files = set(editable_files(work_dir, node_spec)) | set(overrides)
    semantic, raw = [], []
    for rel in sorted(files):
        if rel in overrides:
            text = overrides[rel]
        else:
            text = (Path(work_dir) / rel).read_bytes().decode("utf-8", errors="surrogateescape")
        if text is None:
            continue
        semantic.append([rel, _sha256_text(semantic_form(rel, text))])
        raw.append([rel, hashlib.sha256(text.encode("utf-8", errors="surrogateescape")).hexdigest()])
    return (_sha256_text(json.dumps(semantic, separators=(",", ":"))),
            _sha256_text(json.dumps(raw, separators=(",", ":"))))


# --------------------------------------------------------------------------- duplicate registry


@dataclass
class EvaluatedStates:
    """Every evaluated state of a campaign: normalised diff hashes and code-state hashes."""

    by_diff: dict[str, str] = field(default_factory=dict)
    by_state: dict[str, str] = field(default_factory=dict)

    def register(self, trial_id: str, *, code_state_sha256: str, normalised_diff_sha256: str | None = None) -> None:
        self.by_state.setdefault(code_state_sha256, trial_id)
        if normalised_diff_sha256 is not None:
            self.by_diff.setdefault(normalised_diff_sha256, trial_id)

    def check(self, *, normalised_diff_sha256: str | None, code_state_sha256: str | None) -> dict[str, Any]:
        if normalised_diff_sha256 is not None and normalised_diff_sha256 in self.by_diff:
            return {"duplicate": True, "duplicate_kind": "normalised_diff",
                    "duplicate_of": self.by_diff[normalised_diff_sha256]}
        if code_state_sha256 is not None and code_state_sha256 in self.by_state:
            return {"duplicate": True, "duplicate_kind": "code_state",
                    "duplicate_of": self.by_state[code_state_sha256]}
        return {"duplicate": False, "duplicate_kind": None, "duplicate_of": None}

    def to_dict(self) -> dict[str, Any]:
        return {"by_diff": dict(self.by_diff), "by_state": dict(self.by_state)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluatedStates":
        return cls(by_diff=dict(payload.get("by_diff", {})), by_state=dict(payload.get("by_state", {})))


# --------------------------------------------------------------------------- application


@dataclass
class DiffApplication:
    status: str  # applied | refused | inert | duplicate
    provenance: dict[str, Any]

    @property
    def applied(self) -> bool:
        return self.status == "applied"


def apply_diff(
    work_dir: str | Path,
    diff_text: str,
    node_spec: NodeSpec,
    *,
    evaluated: EvaluatedStates | None = None,
    write: bool = True,
) -> DiffApplication:
    """Apply *diff_text* inside the per-run copy *work_dir*; see the module docstring.

    The copy is changed only when the status is ``applied`` (and *write* is true).
    Refused, inert and duplicate diffs leave it untouched.
    """
    work = Path(work_dir).resolve()
    provenance: dict[str, Any] = {
        "schema": DIFF_PROVENANCE_SCHEMA,
        "diff_text": diff_text,
        "diff_sha256": hashlib.sha256(diff_text.encode("utf-8", errors="surrogateescape")).hexdigest()
        if isinstance(diff_text, str) else None,
        "normalised_diff_sha256": None,
        "files_touched": [],
        "scope": None,
        "hunks": [],
        "incumbent_code_state_sha256": None,
        "incumbent_code_state_raw_sha256": None,
        "code_state_sha256": None,
        "code_state_raw_sha256": None,
        "inert": False,
        "inert_reason": None,
        "duplicate": False,
        "duplicate_kind": None,
        "duplicate_of": None,
        "refusal": None,
        "refusal_detail": None,
    }

    def refused(error: DiffRefused) -> DiffApplication:
        provenance["refusal"] = error.reason
        provenance["refusal_detail"] = str(error)
        return DiffApplication("refused", provenance)

    try:
        incumbent = code_state_hashes(work, node_spec)
        provenance["incumbent_code_state_sha256"], provenance["incumbent_code_state_raw_sha256"] = incumbent
        patches = parse_unified_diff(diff_text)
        provenance["files_touched"] = sorted(p.path for p in patches)
        if not patches:
            provenance["inert"], provenance["inert_reason"] = True, "empty_diff"
            provenance["code_state_sha256"], provenance["code_state_raw_sha256"] = incumbent
            return DiffApplication("inert", provenance)
        provenance["normalised_diff_sha256"] = normalised_diff_sha256(patches)

        scope = validate_edit_scope(provenance["files_touched"], node_spec)
        provenance["scope"] = {"valid": scope.valid, "violations": list(scope.violations)}
        if not scope.valid:
            raise DiffRefused("scope_violation", "; ".join(scope.violations))

        new_contents: dict[str, str | None] = {}
        old_contents: dict[str, str | None] = {}
        for patch in patches:
            target = _safe_target(work, patch.path)
            old_text = _read_existing(target, patch)
            new_text, hunk_log = _apply_patch(old_text, patch)
            old_contents[patch.path] = old_text
            new_contents[patch.path] = new_text
            provenance["hunks"].extend(hunk_log)
    except DiffRefused as error:
        return refused(error)

    resulting = code_state_hashes(work, node_spec, overrides=new_contents)
    provenance["code_state_sha256"], provenance["code_state_raw_sha256"] = resulting

    if all(semantic_form(p, old_contents[p]) == semantic_form(p, new_contents[p]) for p in new_contents):
        changed_raw = any(old_contents[p] != new_contents[p] for p in new_contents)
        provenance["inert"] = True
        provenance["inert_reason"] = "normalised_equal" if changed_raw else "no_change"
        if any(PurePosixPath(p).suffix == ".py" for p in new_contents):
            provenance["inert_reason"] += "_ast_equal_to_incumbent"
        return DiffApplication("inert", provenance)

    if evaluated is not None:
        verdict = evaluated.check(normalised_diff_sha256=provenance["normalised_diff_sha256"],
                                  code_state_sha256=resulting[0])
        provenance.update(verdict)
        if verdict["duplicate"]:
            return DiffApplication("duplicate", provenance)

    if write:
        for rel, text in new_contents.items():
            target = _safe_target(work, rel)
            if text is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8", errors="surrogateescape"))
    return DiffApplication("applied", provenance)


def _safe_target(work: Path, rel: str) -> Path:
    candidate = work / rel
    # No component of the path may be a symlink: a symlinked directory inside an
    # editable path could otherwise reach a frozen path (or leave the copy) while
    # the scope validator sees only the literal path.
    current = work
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            raise DiffRefused("unsafe_path", f"{rel}: {current.relative_to(work).as_posix()} is a symlink")
    resolved = candidate.resolve(strict=False)
    if resolved != work and work not in resolved.parents:
        raise DiffRefused("unsafe_path", f"{rel} resolves outside the working copy")
    return candidate


def _read_existing(target: Path, patch: FilePatch) -> str | None:
    if patch.old_path is None:
        if target.exists():
            raise DiffRefused("context_mismatch", f"{patch.path} is created by the diff but already exists")
        return None
    if not target.is_file():
        raise DiffRefused("context_mismatch", f"{patch.path} does not exist in the working copy")
    try:
        return target.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise DiffRefused("unsupported_diff_feature", f"{patch.path} is not UTF-8 text") from error


def _apply_patch(old_text: str | None, patch: FilePatch) -> tuple[str | None, list[dict[str, Any]]]:
    if old_text is None:
        src: list[str] = []
        trailing_newline = True
    else:
        src = old_text.split("\n")
        trailing_newline = bool(src) and src[-1] == ""
        if trailing_newline:
            src.pop()
    out: list[str] = []
    cursor = 0
    log: list[dict[str, Any]] = []
    for hunk in patch.hunks:
        old_block = hunk.old_block()
        expected = hunk.old_start - 1 if hunk.old_count > 0 else hunk.old_start
        position = None
        if cursor <= expected <= len(src) and src[expected:expected + len(old_block)] == old_block:
            position = expected
        elif old_block:
            hits = [k for k in range(cursor, len(src) - len(old_block) + 1)
                    if src[k:k + len(old_block)] == old_block]
            if len(hits) > 1:
                raise DiffRefused("ambiguous_hunk", f"{patch.path}: hunk @{hunk.old_start} matches {len(hits)} places")
            if hits:
                position = hits[0]
        if position is None:
            raise DiffRefused("context_mismatch", f"{patch.path}: hunk @{hunk.old_start} does not match")
        out.extend(src[cursor:position])
        out.extend(hunk.new_block())
        cursor = position + len(old_block)
        log.append({"file": patch.path, "old_start": hunk.old_start, "applied_at": position + 1,
                    "offset": position - expected})
        if cursor == len(src):  # this hunk reaches the end of the file
            trailing_newline = not hunk.new_no_eol
    out.extend(src[cursor:])
    if patch.new_path is None:
        if out:
            raise DiffRefused("context_mismatch", f"{patch.path}: deletion leaves content behind")
        return None, log
    text = "\n".join(out)
    if out and trailing_newline:
        text += "\n"
    return text, log
