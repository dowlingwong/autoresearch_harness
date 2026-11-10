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
5. **duplicate detection** against an :class:`EvaluatedStates` registry.  The
   **resulting code-state hash** (the same normalisation over every editable
   file of the copy; files that are not UTF-8 are hashed by their bytes) is
   authoritative: a diff is a duplicate if and only if its resulting code
   state was already evaluated.  The normalised diff hash (per hunk, the
   context, removed and added lines in order; line numbers ignored) is
   information only: ``same_diff_hash_other_state`` (same incumbent, same
   diff hash, different resulting code, for example the same edit to another
   copy of a repeated block) and ``seen_diff_other_incumbent`` (the same diff
   evaluated against another incumbent);
5b. **new files** are refused unless the caller allows them
   (``allow_new_files``, from the node spec).  Even then a new file is refused
   if a path component that does not exist yet in the copy, or the new
   module's own name, could shadow an import: ``sitecustomize``,
   ``usercustomize``, a standard-library or built-in module, or a top-level
   module found on a **fixed import path** (the run interpreter's ``sys.path``
   queried from ``/``, without ``''``, the current directory or the copy).
   The basis used is recorded in the provenance (``shadow_check``);
6. **provenance**: the diff text and its hash, the normalised diff hash, the
   files touched, the hunks with any offsets, and the incumbent and resulting
   code-state hashes (normalised and raw).

Nothing here is used by the T0 path (``manager_proposal/v1``).
"""
from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

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
        if line.startswith("new file mode") and line.split()[-1] != "100644":
            raise DiffRefused("unsupported_diff_feature", f"file mode {line.split()[-1]} (only 100644)")
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
    return hashlib.sha256(text.encode("utf-8", errors="surrogateescape")).hexdigest()


_CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")


def _normalise_line(line: str, suffix: str) -> str | None:
    stripped = line.rstrip().expandtabs(4)
    if not stripped.strip():
        return None
    if suffix in _HASH_COMMENT_SUFFIXES and stripped.lstrip().startswith("#"):
        return None
    return stripped


def normalised_text(text: str, suffix: str) -> str:
    """Whitespace- and comment-insensitive form of a non-Python file (or unparsable Python).

    A shebang on line 1 is kept: it selects the interpreter, so it is not a comment.
    """
    lines = text.split("\n")
    out = []
    for index, line in enumerate(lines):
        if index == 0 and line.startswith("#!"):
            out.append(line.rstrip())
            continue
        norm = _normalise_line(line, suffix)
        if norm is not None:
            out.append(norm)
    return "\n".join(out)


def _python_header(text: str) -> str:
    """Shebang and PEP 263 coding lines (lines 1-2), which the AST does not see."""
    head = []
    for index, line in enumerate(text.split("\n")[:2]):
        if (index == 0 and line.startswith("#!")) or _CODING_RE.match(line):
            head.append(line.rstrip())
    return "\n".join(head)


def semantic_form(path: str, text: str | None) -> str:
    """The form compared for inertness and hashed for the code state."""
    if text is None:
        return "<absent>"
    suffix = PurePosixPath(path).suffix
    if suffix == ".py":
        try:
            return "AST:" + _python_header(text) + "\n" + ast.dump(ast.parse(text), include_attributes=False)
        except (SyntaxError, ValueError):
            return "TEXT:" + normalised_text(text, suffix)
    return "TEXT:" + normalised_text(text, suffix)


def semantic_form_bytes(path: str, raw: bytes) -> str:
    """:func:`semantic_form` for file bytes; files that are not UTF-8 are hashed by their bytes."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "BYTES:" + hashlib.sha256(raw).hexdigest()
    return semantic_form(path, text)


def normalised_diff_sha256(patches: Iterable[FilePatch]) -> str:
    """Hash of the diff per file and per hunk: the context, removed and added lines in
    order, each normalised (trailing whitespace, blank lines and comment lines dropped),
    without headers or line numbers.  Context is kept so the same change at two
    different places hashes differently (review T2 round 1, finding 1)."""
    payload = []
    for patch in sorted(patches, key=lambda p: p.path):
        suffix = PurePosixPath(patch.path).suffix
        hunks = []
        for hunk in patch.hunks:
            lines = []
            for tag, text in hunk.lines:
                norm = _normalise_line(text, suffix)
                if norm is not None:
                    lines.append([tag, norm])
            hunks.append(lines)
        kind = "create" if patch.old_path is None else "delete" if patch.new_path is None else "modify"
        payload.append([patch.path, kind, hunks])
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
            if text is None:
                continue
            data = text.encode("utf-8", errors="surrogateescape")
        else:
            data = (Path(work_dir) / rel).read_bytes()
        semantic.append([rel, _sha256_text(semantic_form_bytes(rel, data))])
        raw.append([rel, hashlib.sha256(data).hexdigest()])
    return (_sha256_text(json.dumps(semantic, separators=(",", ":"))),
            _sha256_text(json.dumps(raw, separators=(",", ":"))))


# --------------------------------------------------------------------------- duplicate registry


@dataclass
class EvaluatedStates:
    """Every evaluated state of a campaign.

    - ``by_state``: resulting code-state hash -> first trial id;
    - ``by_diff``: ``"<incumbent code-state hash>:<normalised diff hash>"`` -> first
      trial id, so a diff is a duplicate only against the same incumbent;
    - ``diff_any``: normalised diff hash -> first trial id (any incumbent), reported
      as ``seen_diff_other_incumbent`` and never treated as a duplicate.
    """

    by_diff: dict[str, str] = field(default_factory=dict)
    by_state: dict[str, str] = field(default_factory=dict)
    diff_any: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _key(incumbent: str | None, diff_hash: str) -> str:
        return f"{incumbent or '?'}:{diff_hash}"

    def register(self, trial_id: str, *, code_state_sha256: str, normalised_diff_sha256: str | None = None,
                 incumbent_code_state_sha256: str | None = None) -> None:
        self.by_state.setdefault(code_state_sha256, trial_id)
        if normalised_diff_sha256 is not None:
            self.by_diff.setdefault(self._key(incumbent_code_state_sha256, normalised_diff_sha256), trial_id)
            self.diff_any.setdefault(normalised_diff_sha256, trial_id)

    def check(self, *, normalised_diff_sha256: str | None, code_state_sha256: str | None,
              incumbent_code_state_sha256: str | None = None) -> dict[str, Any]:
        """The resulting code state decides; diff-hash matches are reported as information.

        (Review T2 round 2, finding 1: two different edits with identical context, for
        example ``-U0`` diffs or edits to different copies of a repeated block, share a
        diff hash but reach different code states; they are not duplicates.)
        """
        verdict: dict[str, Any] = {"duplicate": False, "duplicate_kind": None, "duplicate_of": None,
                                   "same_diff_hash_other_state": None, "seen_diff_other_incumbent": None}
        if code_state_sha256 is not None and code_state_sha256 in self.by_state:
            verdict.update(duplicate=True, duplicate_kind="code_state", duplicate_of=self.by_state[code_state_sha256])
        if normalised_diff_sha256 is not None:
            key = self._key(incumbent_code_state_sha256, normalised_diff_sha256)
            if key in self.by_diff:
                if not verdict["duplicate"]:
                    verdict["same_diff_hash_other_state"] = self.by_diff[key]
            elif normalised_diff_sha256 in self.diff_any:
                verdict["seen_diff_other_incumbent"] = self.diff_any[normalised_diff_sha256]
        return verdict

    def to_dict(self) -> dict[str, Any]:
        return {"by_diff": dict(self.by_diff), "by_state": dict(self.by_state), "diff_any": dict(self.diff_any)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluatedStates":
        return cls(by_diff=dict(payload.get("by_diff", {})), by_state=dict(payload.get("by_state", {})),
                   diff_any=dict(payload.get("diff_any", {})))


SHADOW_ALWAYS = frozenset({"sitecustomize", "usercustomize"})


_IMPORT_PATH_CACHE: dict[str, tuple[str, ...]] = {}


def fixed_import_path(interpreter: str = "python3", *, exclude: Iterable[str | Path] = ()) -> tuple[list[str], str]:
    """The run interpreter's ``sys.path`` queried from ``/`` (so no working directory is on
    it), without ``''``, ``/``, the harness's current directory and *exclude*.  Falls back to
    this process's ``sys.path`` with the same exclusions.  Returns (path, basis)."""
    basis = f"interpreter:{interpreter}"
    entries = _IMPORT_PATH_CACHE.get(interpreter)
    if entries is None:
        try:
            out = subprocess.run([interpreter, "-c", "import json, sys; print(json.dumps(sys.path))"],
                                 cwd="/", capture_output=True, text=True, timeout=30, check=True).stdout
            entries = tuple(str(e) for e in json.loads(out))
            _IMPORT_PATH_CACHE[interpreter] = entries
        except (OSError, subprocess.SubprocessError, ValueError):
            entries = tuple(sys.path)
            basis = "harness_sys_path"
    banned = {os.path.realpath(os.getcwd()), os.path.realpath("/")}
    banned_dirs = [os.path.realpath(str(e)) for e in exclude]
    path = []
    for entry in entries:
        if not entry:
            continue
        real = os.path.realpath(entry)
        if real in banned or any(real == d or real.startswith(d + os.sep) for d in banned_dirs):
            continue
        path.append(entry)
    return path, basis


def shadowing_name(rel: str, work_dir: str | Path | None = None, *,
                   import_path: Sequence[str] | None = None) -> str | None:
    """The first name of a new file's path that could newly shadow an import, or ``None``.

    Only path components that do not exist yet in *work_dir* are checked (an existing
    directory cannot newly shadow anything), plus the stem of a new module file.
    """
    pure = PurePosixPath(rel)
    names: list[str] = []
    current = Path(work_dir) if work_dir is not None else None
    for part in pure.parts[:-1]:
        if current is not None:
            current = current / part
            if current.exists():
                continue
        names.append(part)
    if pure.suffix in {".py", ".pyc", ".so", ".pth"} or pure.name == "__init__.py":
        names.append(pure.stem)
    if import_path is None:
        import_path, _ = fixed_import_path(exclude=[work_dir] if work_dir is not None else [])
    for name in names:
        if not name.isidentifier() or name == "__init__":
            continue
        if name in SHADOW_ALWAYS or name in sys.stdlib_module_names or name in sys.builtin_module_names:
            return name
        try:
            if importlib.machinery.PathFinder.find_spec(name, list(import_path)) is not None:
                return name
        except (ImportError, ValueError):
            continue
    return None


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
    allow_new_files: bool = False,
    interpreter: str = "python3",
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
        "same_diff_hash_other_state": None,
        "seen_diff_other_incumbent": None,
        "shadow_check": None,
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
        for patch in patches:
            if patch.old_path is None:
                if not allow_new_files:
                    raise DiffRefused("new_file_not_allowed",
                                      f"{patch.path}: the node does not allow new files (allow_new_files)")
                if provenance["shadow_check"] is None:
                    import_path, basis = fixed_import_path(interpreter, exclude=[work])
                    provenance["shadow_check"] = {"basis": basis, "import_path": import_path,
                                                  "always": sorted(SHADOW_ALWAYS),
                                                  "stdlib_and_builtin": True}
                name = shadowing_name(patch.path, work, import_path=provenance["shadow_check"]["import_path"])
                if name is not None:
                    raise DiffRefused("shadowing_module",
                                      f"{patch.path}: {name!r} could shadow an importable module")

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
                                  code_state_sha256=resulting[0],
                                  incumbent_code_state_sha256=incumbent[0])
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
