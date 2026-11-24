"""A Track G campaign driven by the v2-diff LLM manager, behind an explicit flag (M10; AI-assisted draft).

``run_v2_diff_campaign(..., enable_v2_diff=True)`` is the campaign path for ``manager_proposal/v2-diff``.
Without ``enable_v2_diff=True`` it refuses to run: nothing in the T0 path calls it, and the v1 campaign
code (``autoresearch.control_plane``) is untouched, so the T0 ``manager_proposal/v1`` path stays
byte-identical (``tests/trackg/test_trackg_t0_identity.py``).

Per step (``budget`` manager calls, each charged):

1. the manager proposes one diff against the incumbent; a reply that does not parse is an
   ``invalid_proposal`` (charged, nothing runs);
2. the diff runs in a fresh working copy through :class:`WorkingCopyRunner` (scope, inert and duplicate
   checks, limits, outcome class). A diff that is refused, inert or a duplicate is not run;
3. outcome handling:
   * ``ok``: kept if strictly better than the incumbent on the validation metric (the next incumbent is
     the run's pre-run code snapshot), else discarded;
   * ``divergent``: discarded.  Since R1-TRACKG-PORT-DECISIONS (4) (17 Nov 2026) this includes a run that
     exceeded the **measured** timeout (census baseline × 3): reason ``timeout``, registered as evaluated by
     the runner, never rerun (a timeout is the proposal's failure, not an infrastructure retry);
   * ``infrastructure_failure`` with reason ``timeout`` -- only under a **provisional** timeout (the census's
     measuring run): the same code state is rerun. **Repeated-timeout cap** (author's answer to M10 §5
     question 8, option (a)): after ``timeout_cap`` (2) timeouts of the same code state, the proposal is
     ``divergent`` with reason ``timeout_repeated``, the code state is registered as evaluated, and it is
     never rerun;
   * any other ``infrastructure_failure`` (launch error, kill, an infrastructure log pattern such as Slurm's
     ``DUE TO TIME LIMIT``): rerun up to ``max_infra_reruns`` times, then recorded as
     ``infrastructure_failure_final`` (not registered as evaluated, so it can be proposed and run again later,
     bounded by the manager budget). These reruns never count toward the timeout cap.
   Reruns of the same proposal do not consume manager calls. The timeout count is per code state over the
   whole campaign: a code state that timed out once in an earlier step reaches the cap on its next timeout.
   A diff seen before against another incumbent reaches a new code state and is run
   (``seen_diff_other_incumbent`` is recorded; M10 §5 question 7, option (a)).

The baseline gets the same rerun and cap handling; if it still has no ``ok`` run, the campaign writes a
``baseline_failed`` ledger line and STOPs (nothing is compared against a missing incumbent value).  Under a
measured timeout a baseline timeout is ``divergent`` and therefore such a STOP at once: the census measured that
baseline's run time, so a timeout means the host or the node changed.

R1-HARNESS-ISOLATION: the campaign refuses a node without an ``isolation`` block unless ``unisolated_reason`` is
given (never for T1 label runs or RL training).  It pins the frozen files from the node kit once at the start
and passes the pin, the kit (read-only for the worker) and the incumbent's expected code state to every run; a
working copy that differs (something outside the declared diffs changed the kit or an incumbent snapshot)
writes a ``stop_integrity`` ledger line and STOPs. A transport
failure of the manager (``ManagerTransportStop``) writes a ``stop_transport`` line and STOPs. ``record_gate`` (optional; the T1 jobs pass ``noise_floor.check_effective``) is called on every executed run
record; a refusal writes a ``stop_unisolated`` line and STOPs.  ``runner_factory`` (optional; default
:class:`WorkingCopyRunner`, so G.4's runner is unchanged; the hunk-header recount below applies to G.4 too) builds the runner of every run with the same arguments; the
G.5 engine (:mod:`autoresearch.trackg.audit_g5`) passes a subclass that adds the code state of a not-run diff to the
top level of its record.  The campaign has no
resume: an HTCondor eviction of the job that runs it ends the campaign, and a restart refuses the existing
``runs_root`` (Unit 2's resume-safe units are not wired here).

Every step appends one line to ``<runs_root>/campaign_ledger.jsonl`` (never overwritten): the proposal
fields, the manager call record (prompt hash, seed, raw reply and its hash, usage), every run record of
the step (``manager_view``: no sealed test metric), the decision and the incumbent after the step.

**Hunk-header recount** (R1-G5-DIFF-RECOUNT, option A; :func:`recount_hunk_headers`).  Before a proposal's diff
goes to the runner, the line counts of its hunk headers are recomputed from the hunk bodies (``git apply
--recount`` semantics, with the differences listed in that function).  Only the counts in ``@@`` lines of a
**copy** are rewritten; ``diff_support`` (the strict parser, scope, context and offset matching, inert and
duplicate checks) is unchanged and applies the copy.  The ledger keeps the manager's text in ``proposal.diff``
(D7-DIFF B, O1); the step's ``diff_recount`` block discloses whether, and which, headers were rewritten, with the
sha256 of both texts.  The run record's ``diff.diff_text`` and ``diff.diff_sha256`` are those of the text the runner
applied (the copy; equal to the manager's text when nothing was recounted).  The G.5 audit re-materialises units
through the same function (:mod:`autoresearch.trackg.audit_g5`).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from autoresearch.trackg.diff_support import _ALLOWED_GIT_HEADERS, _HUNK_RE, _REFUSED_GIT_HEADERS, EvaluatedStates
from autoresearch.trackg.manager_v2_diff import ManagerTransportStop, V2DiffManager
from autoresearch.trackg.node_template import TrackGNodeSpec
from autoresearch.trackg.outcome import DIVERGENT, INFRASTRUCTURE_FAILURE, OK, RunOutcome
from autoresearch.trackg import isolation
from autoresearch.trackg.runner import IntegrityError, ResourceLimits, WorkingCopyRunner
from autoresearch.trackg.selection import manager_view

LEDGER_SCHEMA = "trackg_v2_diff_campaign_ledger/v1"
TIMEOUT_REASON = "timeout"
TIMEOUT_REPEATED = "timeout_repeated"


class V2DiffCampaignError(RuntimeError):
    pass


# =============================================================================================== hunk-header recount
DIFF_RECOUNT_RULE = "trackg_diff_recount/v1"
_FILE_LEVEL_PREFIXES = _ALLOWED_GIT_HEADERS + _REFUSED_GIT_HEADERS   # lines no hunk body line can start with


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogateescape")).hexdigest()


@dataclass
class DiffRecount:
    """The result of :func:`recount_hunk_headers`: ``text`` is what the runner applies."""

    text: str
    hunks: int = 0                                    # hunk headers the scan reached
    recounted: list[dict[str, Any]] = field(default_factory=list)
    left: str | None = None                           # why a hunk was left to the strict parser (scan stopped)
    left_line: int | None = None

    @property
    def changed(self) -> bool:
        return bool(self.recounted)

    def record(self, raw: str) -> dict[str, Any]:
        """The ledger's ``diff_recount`` block (no diff text)."""
        return {"rule": DIFF_RECOUNT_RULE, "changed": self.changed, "hunks_seen": self.hunks,
                "hunks_recounted": len(self.recounted), "recounted": list(self.recounted),
                "left_to_strict_parser": self.left, "left_at_line": self.left_line,
                "raw_diff_sha256": _sha(raw), "applied_diff_sha256": _sha(self.text)}


def _strict_end(lines: list[str], i: int, old_count: int, new_count: int) -> int | None:
    """Where ``diff_support._parse_hunk`` would end a hunk read with the header's counts (the index after it), or
    None where it would refuse (a bad line tag, or the counts not met before the end).  Same loop as the parser."""
    old_seen = new_seen = 0
    while i < len(lines) and (old_seen < old_count or new_seen < new_count):
        line = lines[i]
        if line.startswith("\\"):
            i += 1
            continue
        tag = line[0] if line else " "
        if tag not in " -+":
            return None
        old_seen += tag in " -"
        new_seen += tag in " +"
        i += 1
    if old_seen != old_count or new_seen != new_count:
        return None
    if i < len(lines) and lines[i].startswith("\\"):
        i += 1
    return i


def _separator(line: str) -> bool:
    """A blank line (empty or whitespace only, a line of one space included): what the strict parser skips between
    file patches."""
    return line.strip() == ""


def _anchored(body: list[str]) -> bool:
    """A non-blank old line (context or removed) among hunk body lines."""
    return any(line[:1] in (" ", "-") and line[1:].strip() for line in body)


def _blank_placed(body: list[str]) -> bool:
    """Old lines that are all blank: empty context lines and no non-blank context or removed line."""
    return not _anchored(body) and any(line == "" or (line[:1] == " " and not line[1:].strip()) for line in body)


def _is_pair(lines: list[str], i: int) -> bool:
    return lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ ")


def _strict_reading_fits(lines: list[str], end: int) -> bool:
    """The header's counts describe the hunk: what follows its strict reading is not more hunk body.  That is the
    end, a hunk header, or (after blank lines, which the strict parser skips between file patches) the end, a
    ``---``/``+++`` file-header pair or any line that is not a body line other than ``@@`` (the strict parser then
    decides as before).  Blank lines followed by ``@@`` are more body (empty context, as git counts them); the strict
    parser refuses that layout, so this changes no diff it accepts."""
    if end >= len(lines) or lines[end].startswith("@@"):
        return True
    j = end
    while j < len(lines) and lines[j].strip() == "":
        j += 1
    if j >= len(lines) or _is_pair(lines, j):
        return True
    if lines[j].startswith("@@"):      # blank lines and then a hunk header: the blank lines are empty context
        return False
    return lines[j][0] not in " -+"


def recount_hunk_headers(diff_text: Any) -> DiffRecount:
    """Recompute each hunk header's line counts from its body (``git apply --recount`` semantics) in a copy.

    The scan follows ``diff_support.parse_unified_diff``'s structure.  Per hunk:

    1. **The header first.**  If the header's counts give the strict parser's reading and what follows is not more
       body (:func:`_strict_reading_fits`), the hunk is left as it is.  So every diff the strict parser accepts is
       returned unchanged (byte for byte), and so are its refusals that do not come from wrong counts.
    2. **Otherwise the body decides.**  The body runs from the line after the header to the first of: a line that
       starts with ``@@`` (the next hunk header, parsed or refused by the strict parser); a file-level git line
       (``diff --git``, ``index``, ``new file mode``, ``deleted file mode``, or a refused rename/copy/mode/binary
       line); in a file patch **without** a ``diff --git`` line, a ``--- `` line immediately followed by a ``+++ ``
       line (the next file header); the end of the diff.  **Blank lines directly before the next file header or
       the end** (empty or whitespace only, a line of one space included) **are separators**, as the strict parser
       skips them between file patches.  Blank lines elsewhere (before an ``@@`` line, or inside the body) are empty
       context, as in git.  This differs from git's recount, which counts the trailing blank lines as context:
       where a real trailing empty context line is dropped, a hunk that git refuses may be accepted here, with one
       context line fewer, placed by its other old lines (a hunk left with no non-blank old line is refused, see
       point 3).  Counting them instead moved hunks to other blank lines of the file (review confirmation round),
       so it is not done.  Its old count is the context plus removed (``-``) lines, its new count the context plus added
       (``+``) lines; ``\\ No newline at end of file`` markers are not counted.  The header's start lines and its
       trailing text are kept; only the two counts are rewritten (as ``start,count``).
    3. **No safe recount: the hunk is left as it is**, the scan stops, and the strict parser refuses the diff
       (``malformed_hunk`` or ``malformed_diff``).  In a diff with several hunks, earlier hunks may already be
       recounted, so the strict parser can reach a fault it did not reach before and the reason can differ from
       G.4's (for example ``malformed_hunk`` where G.4 said ``malformed_diff``); the diff is refused either way:

       * a body line with a bad tag (a line that is none of the above and does not start with `` ``, ``-``,
         ``+`` or ``\\``);
       * an empty body (no counted line);
       * a body without old lines whose header has old lines: the header's start line alone would place the
         insertion, and ``diff_support`` uses a different convention for an old count of 0, so the position would
         change (``no_old_lines``);
       * a body with no non-blank old line (no non-blank context and no removed line) that either has blank context
         lines or lost trailing blank lines as separators (``blank_only_anchor``).  Such a hunk is placed only by its
         header's start line and by blank lines, which match many places of a file; whether a trailing blank line
         was context or a separator then decides the place (it changes the old count, and with it where
         ``diff_support`` looks first), and nothing in the text decides which.  A pure insertion with no blank line
         at all keeps the header's own convention and is recounted (only its new count changes);
       * in a file patch without a ``diff --git`` line, a ``--- ``/``+++ `` pair inside the body that is not
         immediately followed by an ``@@`` line: it is either a removed ``-- ...`` line followed by an added
         ``++ ...`` line, or a file header without hunks, and nothing decides which.

    A hunk header that does not match the strict parser's pattern stops the scan too (``malformed_hunk`` follows),
    as does any line the strict parser refuses between file patches.

    **A recounted diff keeps no blank-context-only hunk and none followed by a separator.**  If any hunk of the diff
    is recounted, every hunk the header decides (point 1) must also have a non-blank old line, or be a pure insertion
    with no blank line in or right after it.  (A header-fitting hunk whose only old lines are removed blank lines, or
    an anchored one whose fitting header counts a trailing separator as context, is not flagged; it is read exactly as
    the strict parser reads it.)  Otherwise the whole diff is left as written (``blank_only_anchor``) and the strict parser refuses it, as in
    G.4: a fitting header that counts a trailing blank line as context places such a hunk at a blank line of the file,
    which need not be where the manager meant.  A diff the strict parser accepts recounts no hunk, so this changes
    none of them.

    **The ``-- ``/``++ `` rule.**  Where the patch has a ``diff --git`` line, a ``--- ``/``+++ `` pair inside a
    recounted body is body (a removed line, then an added line), as in git, whose recount ends a body only at
    ``@@ `` or ``diff ``; if it was in fact a second file's header without its ``diff --git`` line, the removed
    line ``-- a/<path>`` must then match the first file exactly, so the diff is refused (``context_mismatch``).
    Without a ``diff --git`` line (the prompt asks for ``--- a/``/``+++ b/`` headers), such a pair followed by
    ``@@`` is a file header; the strict parser then checks it as one (paths, no rename, a file only once).
    **Disclosed residual:** if such a pair was meant as body (a removed line ``-- a/P`` and an added line
    ``++ b/P``, where ``P`` is another editable file), the hunks after it are applied to ``P`` whenever their context
    matches ``P`` exactly; git's recount would have kept them in the current file.  No text-only test separates the
    two readings without also refusing every wrong-count multi-file patch in the prompt's format, so this is not
    refused (``tests/trackg/test_trackg_diff_recount.py`` builds such a case).

    Nothing else changes: the strict parser, the scope check, exact context matching with its unique-offset
    search, and the inert and duplicate checks all run on the copy as before.  Not text: returned as it is."""
    if not isinstance(diff_text, str):
        return DiffRecount(text=diff_text)
    rewritten = diff_text.split("\n")
    lines = rewritten[:-1] if rewritten and rewritten[-1] == "" else list(rewritten)   # as the parser drops it
    n = len(lines)
    out = DiffRecount(text=diff_text)

    def stop(reason: str, index: int) -> DiffRecount:
        out.left, out.left_line = reason, index + 1
        return finish()

    weak: list[int] = []          # hunks the header decides (point 1) that are placed only by blank lines

    def finish() -> DiffRecount:
        if out.recounted and weak:  # a recounted diff with such a hunk: left as written (the strict parser refuses)
            out.recounted, out.left, out.left_line = [], "blank_only_anchor", weak[0] + 1
        if out.recounted:
            out.text = "\n".join(rewritten)
        return out

    i, git_pending = 0, False
    while i < n:
        line = lines[i]
        if line.startswith(_REFUSED_GIT_HEADERS) or (line.startswith("new file mode")
                                                       and line.split()[-1] != "100644"):
            return stop("unsupported_file_line", i)
        if line.strip() == "" or line.startswith(_ALLOWED_GIT_HEADERS):
            git_pending = git_pending or line.startswith("diff --git ")
            i += 1
            continue
        if not _is_pair(lines, i):
            return stop("not_a_file_header", i)
        git_style, git_pending = git_pending, False
        i += 2
        while i < n and lines[i].startswith("@@"):
            match = _HUNK_RE.match(lines[i])
            if not match:
                return stop("bad_hunk_header", i)
            out.hunks += 1
            h_old = int(match.group(2)) if match.group(2) is not None else 1
            h_new = int(match.group(4)) if match.group(4) is not None else 1
            end = _strict_end(lines, i + 1, h_old, h_new)
            if end is not None and _strict_reading_fits(lines, end):
                if _blank_placed(lines[i + 1:end]) or (not _anchored(lines[i + 1:end])
                                                        and end < n and lines[end].strip() == ""):
                    weak.append(i)
                i = end
                continue
            # the body decides
            j, terminator = i + 1, "end"
            while j < n:
                body = lines[j]
                if body.startswith("@@"):
                    terminator = "hunk"
                    break
                if body.startswith(_FILE_LEVEL_PREFIXES):
                    terminator = "file"
                    break
                if not git_style and _is_pair(lines, j):
                    if j + 2 >= n or not lines[j + 2].startswith("@@"):
                        return stop("ambiguous_file_header_pair", j)
                    terminator = "file"
                    break
                if body.strip() and body[0] not in " -+\\":
                    return stop("bad_line_tag", j)
                j += 1
            stop_at = j
            if terminator != "hunk":       # blank lines before the next file header or the end: separators
                while stop_at > i + 1 and _separator(lines[stop_at - 1]):
                    stop_at -= 1
            trimmed = stop_at < j
            b_old = b_new = 0
            anchored = False               # a non-blank old line (context or removed)
            for k in range(i + 1, stop_at):
                body = lines[k]
                if body.startswith("\\"):
                    continue
                tag = body[0] if body else " "
                if tag not in " -+":       # a whitespace-only line such as a tab: a bad tag for the strict parser
                    return stop("bad_line_tag", k)
                b_old += tag in " -"
                b_new += tag in " +"
                anchored = anchored or (tag in " -" and body[1:].strip() != "")
            if b_old + b_new == 0:
                return stop("empty_body", i)
            if b_old == 0 and h_old != 0:
                return stop("no_old_lines", i)
            if not anchored and (b_old > 0 or trimmed):
                return stop("blank_only_anchor", i)
            if (b_old, b_new) != (h_old, h_new):
                rewritten[i] = (f"@@ -{match.group(1)},{b_old} +{match.group(3)},{b_new} @@"
                                + lines[i][match.end():])
                out.recounted.append({"line": i + 1, "header": [h_old, h_new], "body": [b_old, b_new]})
            i = stop_at
    return finish()


def run_v2_diff_campaign(
    spec: TrackGNodeSpec,
    node_dir: str | Path,
    runs_root: str | Path,
    manager: V2DiffManager,
    limits: ResourceLimits,
    budget: int,
    *,
    enable_v2_diff: bool = False,
    timeout_cap: int = 2,
    max_infra_reruns: int = 2,
    train_seed: int | None = None,
    split_seed: int | None = None,
    interpreter: str = "python3",
    unisolated_reason: str | None = None,
    record_gate: Callable[[dict[str, Any]], None] | None = None,
    runner_factory: Callable[..., WorkingCopyRunner] = WorkingCopyRunner,
) -> list[dict[str, Any]]:
    if enable_v2_diff is not True:
        raise V2DiffCampaignError("the v2-diff campaign path is off: pass enable_v2_diff=True explicitly")
    if spec.isolation is None and not unisolated_reason:
        raise V2DiffCampaignError("R1-HARNESS-ISOLATION: the node is not isolated (no isolation block); a campaign "
                                  "on it needs an explicit unisolated_reason (never for T1 label runs or RL)")
    if type(budget) is not int or budget < 1:
        raise V2DiffCampaignError("budget must be a positive integer")
    if type(timeout_cap) is not int or timeout_cap < 1 or type(max_infra_reruns) is not int or max_infra_reruns < 0:
        raise V2DiffCampaignError("timeout_cap must be >= 1 and max_infra_reruns >= 0")
    runs = Path(runs_root).resolve()
    runs.mkdir(parents=True, exist_ok=False)
    ledger = runs / "campaign_ledger.jsonl"
    ledger.touch(exist_ok=False)
    evaluated = EvaluatedStates()
    timeouts: dict[str, int] = {}
    kit_dir = Path(node_dir).resolve()
    incumbent_dir = kit_dir
    # Pinned once from the kit: every run's working copy (incumbent snapshots included) must match it.
    frozen_pin = isolation.frozen_file_hashes(kit_dir, spec.frozen_paths)
    expected_state: list[str | None] = [None]   # the incumbent's raw code state after a kept step
    history: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []

    def append(entry: dict[str, Any]) -> None:
        entries.append(entry)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            handle.flush()

    def better(value: float, best: float | None) -> bool:
        return best is None or (value > best if spec.metric_direction == "maximize" else value < best)

    def run_with_retries(step_tag: str, diff_text: str | None, entry: dict[str, Any]) -> Any:
        """Run (and rerun) one code state; returns the final outcome or None if the diff was not run.

        Sets entry["decision_hint"]: "not_run_<status>", "timeout_repeated", "infrastructure_failure_final" or "ran"."""
        attempt, reruns_other = 0, 0
        while True:
            run_id = step_tag + (f"_r{attempt}" if attempt else "")
            try:
                result = runner_factory(spec, incumbent_dir, runs, frozen_pin=frozen_pin,
                                         protect_paths=(kit_dir,)).run(
                    run_id, limits, train_seed=train_seed, split_seed=split_seed, diff_text=diff_text,
                    evaluated=evaluated, interpreter=interpreter, expected_incumbent_state=expected_state[0])
            except IntegrityError as error:
                append({"schema": LEDGER_SCHEMA, "step": entry.get("step"), "decision": "stop_integrity",
                        "error": str(error), "runs": entry["runs"], "incumbent_after": str(incumbent_dir)})
                raise
            entry["runs"].append(manager_view(result.record))
            if record_gate is not None and result.record.get("executed"):
                try:
                    record_gate(result.record)
                except Exception as error:  # the gate's refusal: the run is not usable, the campaign STOPs
                    append({"schema": LEDGER_SCHEMA, "step": entry.get("step"), "decision": "stop_unisolated",
                            "error": f"{type(error).__name__}: {error}", "runs": entry["runs"],
                            "incumbent_after": str(incumbent_dir)})
                    raise V2DiffCampaignError(f"STOP: run {run_id} failed the record gate: {error}") from error
            outcome = result.outcome
            if outcome is None:
                entry["decision_hint"] = f"not_run_{result.record['diff']['status']}"
                return None, result
            if outcome.kind == INFRASTRUCTURE_FAILURE and outcome.reason == TIMEOUT_REASON:
                state = result.record["code_state_sha256"]
                timeouts[state] = timeouts.get(state, 0) + 1
                if timeouts[state] >= timeout_cap:
                    final = RunOutcome.divergent(TIMEOUT_REPEATED, timeouts=timeouts[state], code_state_sha256=state)
                    diff = result.record.get("diff") or {}
                    evaluated.register(run_id, code_state_sha256=state,
                                       normalised_diff_sha256=diff.get("normalised_diff_sha256"),
                                       incumbent_code_state_sha256=result.record.get("incumbent_code_state_sha256"))
                    entry["decision_hint"] = "timeout_repeated"
                    return final, result
                attempt += 1
                continue
            if outcome.kind == INFRASTRUCTURE_FAILURE:
                if reruns_other < max_infra_reruns:
                    reruns_other += 1
                    attempt += 1
                    continue
                entry["decision_hint"] = "infrastructure_failure_final"
                return outcome, result
            entry["decision_hint"] = "ran"
            return outcome, result

    base_entry: dict[str, Any] = {"schema": LEDGER_SCHEMA, "step": 0, "runs": []}
    base_outcome, base_result = run_with_retries("s000_baseline", None, base_entry)
    base_entry.pop("decision_hint", None)
    if base_outcome is None or base_outcome.kind != OK:
        base_entry.update(decision="baseline_failed", final_outcome=base_outcome.to_dict() if base_outcome else None,
                          incumbent_after=str(incumbent_dir), best_after=None)
        append(base_entry)
        raise V2DiffCampaignError("STOP: the baseline did not give an ok run after reruns; nothing is compared "
                                  "against a missing incumbent value")
    best, incumbent_dir = base_outcome.value, base_result.code_dir
    expected_state[0] = base_result.record["code_state_raw_sha256"]
    base_entry.update(decision="baseline", final_outcome=base_outcome.to_dict(),
                      incumbent_after=str(incumbent_dir), best_after=best)
    append(base_entry)

    for step in range(1, budget + 1):
        try:
            proposal, call = manager.propose(spec, incumbent_dir, history, step)
        except ManagerTransportStop as stop:
            append({"schema": LEDGER_SCHEMA, "step": step, "decision": "stop_transport", "error": str(stop),
                    "manager_call": getattr(stop, "call", None), "runs": [],
                    "incumbent_after": str(incumbent_dir), "best_after": best})
            raise
        entry: dict[str, Any] = {"schema": LEDGER_SCHEMA, "step": step, "manager_call": call, "runs": []}
        if proposal is None:
            entry["decision"] = "invalid_proposal"
            summary = None
        else:
            summary = proposal.summary
            entry["proposal"] = {"schema_version": proposal.schema_version, "rationale": proposal.rationale,
                                 "summary": proposal.summary, "objective": proposal.objective,
                                 "diff": proposal.diff}       # the manager's text as written (O1)
            recount = recount_hunk_headers(proposal.diff)
            entry["diff_recount"] = recount.record(proposal.diff)
            outcome, result = run_with_retries(f"s{step:03d}", recount.text, entry)
            hint = entry.pop("decision_hint")
            if outcome is not None:
                entry["final_outcome"] = outcome.to_dict()
            if hint.startswith("not_run_") or hint == "infrastructure_failure_final":
                entry["decision"] = hint
            elif hint == "timeout_repeated" or outcome.kind == DIVERGENT:
                entry["decision"] = "discarded_divergent"
            elif better(outcome.value, best):
                best, incumbent_dir = outcome.value, result.code_dir
                expected_state[0] = result.record["code_state_raw_sha256"]
                entry["decision"] = "kept"
            else:
                entry["decision"] = "discarded"
        entry["incumbent_after"], entry["best_after"] = str(incumbent_dir), best
        metric = (entry.get("final_outcome") or {}).get("value")
        history.append({"step": step, "summary": summary, "decision": entry["decision"], "metric": metric})
        append(entry)
    return entries
