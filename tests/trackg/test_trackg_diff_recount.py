"""R1-G5-DIFF-RECOUNT: hunk-header recount before diff application (``campaign_v2_diff.recount_hunk_headers``).

The recount rewrites only the counts in the ``@@`` lines of a copy; ``diff_support`` (unchanged, P5) applies the copy.
These tests check: wrong counts are accepted and give the same result as the correctly counted diff; context and
offset matching are as strict as before; malformed diffs stay refused with their reasons; the ``-- ``/``++ `` cases;
several hunks and files; "No newline at end of file" markers; that every diff the strict parser accepts is returned
byte-identical; the campaign ledger (O1) and the audit's re-materialisation."""
from __future__ import annotations

import hashlib
import json
import random
import shutil

import pytest

from autoresearch.trackg import audit_g5
from autoresearch.trackg.campaign_v2_diff import (DIFF_RECOUNT_RULE, recount_hunk_headers, run_v2_diff_campaign)
from autoresearch.trackg.diff_support import (DiffRefused, apply_diff, code_state_hashes, normalised_diff_sha256,
                                              parse_unified_diff)
from autoresearch.trackg.manager_v2_diff import V2DiffManager
from autoresearch.trackg.node_template import TrackGNodeSpec
from autoresearch.trackg.runner import ResourceLimits

from .fixtures import make_node, spec_mapping

LIMITS = ResourceLimits(uncapped_reason="test fixture", measured_run_seconds=10)
LONG = [f"V{k:02d} = {k}" for k in range(1, 61)]          # model/long.py: 60 distinct lines
SQL = ["-- header comment", "SELECT 1;", "-- a/q.sql", "SELECT 2;", "++ not python", "SELECT 3;"]


@pytest.fixture()
def env(tmp_path):
    node = make_node(tmp_path)
    (node / "model" / "long.py").write_text("\n".join(LONG) + "\n", encoding="utf-8")
    (node / "model" / "q.sql").write_text("\n".join(SQL) + "\n", encoding="utf-8")
    (node / "model" / "tail.txt").write_text("alpha\nbeta\ngamma", encoding="utf-8")   # no final newline
    spec = TrackGNodeSpec.from_mapping(spec_mapping())
    return node, spec.to_base_node_spec(), spec


def copy(node, tmp_path, name):
    target = tmp_path / name
    shutil.copytree(node, target, symlinks=True)
    return target


def apply_both(node, base, text, tmp_path, name="w"):
    """(application of the raw text, application of the recounted copy), each on its own copy of the node."""
    raw = apply_diff(copy(node, tmp_path, name + "_raw"), text, base, write=False)
    rec = apply_diff(copy(node, tmp_path, name + "_rec"), recount_hunk_headers(text).text, base, write=False)
    return raw, rec


def strict_ok(text):
    try:
        parse_unified_diff(text)
        return True
    except DiffRefused:
        return False


def same_application(a, b):
    """Two applications that agree on everything the recount must not change."""
    keys = ("refusal", "code_state_sha256", "code_state_raw_sha256", "normalised_diff_sha256", "hunks",
            "files_touched", "inert", "inert_reason")
    return a.status == b.status and all(a.provenance[k] == b.provenance[k] for k in keys)


def long_hunk(old_count, new_count, start=21):
    """The G.4 report's shape: 17 old lines (15 context + 2 removed) and 21 new (15 context + 6 added)."""
    body = [" " + x for x in LONG[start - 1:start + 6]]
    body += ["-" + LONG[start + 6], "-" + LONG[start + 7]]
    body += ["+NEW_A = 1", "+NEW_B = 2", "+NEW_C = 3", "+NEW_D = 4", "+NEW_E = 5", "+NEW_F = 6"]
    body += [" " + x for x in LONG[start + 8:start + 16]]
    return (f"--- a/model/long.py\n+++ b/model/long.py\n@@ -{start},{old_count} +{start},{new_count} @@ section\n"
            + "\n".join(body) + "\n")


# ----------------------------------------------------------------------------------------- wrong counts accepted
def test_report_example_wrong_counts_are_accepted_with_the_same_result(env, tmp_path):
    node, base, _ = env
    wrong, right = long_hunk(18, 25), long_hunk(17, 21)
    assert strict_ok(right) and not strict_ok(wrong)
    rec = recount_hunk_headers(wrong)
    assert rec.text == right                                  # only the counts changed; the trailing text is kept
    assert rec.recounted == [{"line": 3, "header": [18, 25], "body": [17, 21]}]
    raw, applied = apply_both(node, base, wrong, tmp_path)
    assert raw.status == "refused" and raw.provenance["refusal"] == "malformed_hunk"     # the G.4 refusal
    good = apply_diff(copy(node, tmp_path, "good"), right, base, write=False)
    assert applied.status == "applied" and same_application(applied, good)
    # the code state is the files' hash: identical to the correctly counted diff's; diff_sha256 is of the text given
    assert applied.provenance["code_state_sha256"] == good.provenance["code_state_sha256"]
    assert applied.provenance["diff_sha256"] == hashlib.sha256(right.encode()).hexdigest()


@pytest.mark.parametrize("old, new", [(17, 20), (5, 5), (30, 40), (1, 1), (16, 21), (17, 22)])
def test_any_wrong_counts_recount_to_the_body(env, tmp_path, old, new):
    node, base, _ = env
    rec = recount_hunk_headers(long_hunk(old, new))
    assert rec.text == long_hunk(17, 21)
    _, applied = apply_both(node, base, long_hunk(old, new), tmp_path)
    assert applied.status == "applied"


def test_counts_too_small_were_malformed_diff_and_are_now_accepted(env, tmp_path):
    """A header smaller than its body made the strict parser stop early and refuse the rest (malformed_diff)."""
    node, base, _ = env
    raw, applied = apply_both(node, base, long_hunk(7, 7), tmp_path)
    assert raw.provenance["refusal"] == "malformed_diff" and applied.status == "applied"


def test_omitted_counts_and_header_text_are_kept(env):
    text = "--- a/model/long.py\n+++ b/model/long.py\n@@ -3 +3 @@ def f():\n V03 = 3\n-V04 = 4\n+V04 = 40\n"
    assert recount_hunk_headers(text).text.splitlines()[2] == "@@ -3,2 +3,2 @@ def f():"


# ----------------------------------------------------------------------------------------- matching is not loosened
def test_wrong_context_is_refused(env, tmp_path):
    node, base, _ = env
    bad = long_hunk(18, 25).replace(" V25 = 25", " V25 = 2500")
    raw, applied = apply_both(node, base, bad, tmp_path)
    assert raw.provenance["refusal"] == "malformed_hunk"
    assert applied.status == "refused" and applied.provenance["refusal"] == "context_mismatch"


def test_wrong_removed_line_is_refused(env, tmp_path):
    node, base, _ = env
    _, applied = apply_both(node, base, long_hunk(18, 25).replace("-V28 = 28", "-V28 = 99"), tmp_path)
    assert applied.provenance["refusal"] == "context_mismatch"


@pytest.mark.parametrize("start", [21, 25, 1, 50])
def test_offsets_and_ambiguity_match_the_correctly_counted_diff(env, tmp_path, start):
    """A wrong start line: the recounted diff is found (or refused) exactly as the correctly counted one."""
    node, base, _ = env
    right = long_hunk(17, 21).replace("@@ -21,17 +21,21 @@", f"@@ -{start},17 +{start},21 @@")
    wrong = long_hunk(18, 25).replace("@@ -21,18 +21,25 @@", f"@@ -{start},18 +{start},25 @@")
    good = apply_diff(copy(node, tmp_path, "good"), right, base, write=False)
    _, applied = apply_both(node, base, wrong, tmp_path)
    assert same_application(applied, good)
    assert applied.provenance["hunks"][0]["offset"] == 21 - start


def test_an_ambiguous_hunk_stays_ambiguous(env, tmp_path):
    node, base, _ = env
    (node / "model" / "rep.py").write_text("A = 1\nB = 2\nC = 3\nA = 1\nB = 2\nC = 3\n", encoding="utf-8")
    right = "--- a/model/rep.py\n+++ b/model/rep.py\n@@ -40,2 +40,2 @@\n A = 1\n-B = 2\n+B = 3\n"
    wrong = right.replace("@@ -40,2 +40,2 @@", "@@ -40,5 +40,9 @@")
    good = apply_diff(copy(node, tmp_path, "good"), right, base, write=False)
    _, applied = apply_both(node, base, wrong, tmp_path)
    assert good.provenance["refusal"] == "ambiguous_hunk" and same_application(applied, good)


# ----------------------------------------------------------------------------------------- malformed stays refused
MALFORMED = [
    ("--- a/model/long.py\n+++ b/model/long.py\n@@ -a,1 +b,1 @@\n-V01 = 1\n+V01 = 2\n", "malformed_hunk"),
    ("--- a/model/long.py\n+++ b/model/long.py\n@@ @@\n-V01 = 1\n+V01 = 2\n", "malformed_hunk"),
    ("--- a/model/long.py\n+++ b/model/long.py\n@@ -1,5 +1,5 @@\n-V01 = 1\n*V01 = 2\n", "malformed_hunk"),
    ("--- a/model/long.py\n+++ b/model/long.py\n@@ -1,5 +1,5 @@\n V01 = 1\n\tV02 = 2\n", "malformed_hunk"),
    ("--- a/../evil.py\n+++ b/../evil.py\n@@ -1,5 +1,5 @@\n-x\n+y\n", "unsafe_path"),
    ("--- /etc/passwd\n+++ /etc/passwd\n@@ -1,5 +1,5 @@\n-x\n+y\n", "unsafe_path"),
    ('--- "a/model/x y.py"\n+++ "b/model/x y.py"\n@@ -1,5 +1,5 @@\n-x\n+y\n', "unsafe_path"),
    ("--- a/model/long.py\n+++ b/model/long.py\n", "malformed_diff"),
    ("--- a/model/long.py\n+++ b/model/other.py\n@@ -1,5 +1,5 @@\n-V01 = 1\n+V01 = 2\n", "unsupported_diff_feature"),
    ("diff --git a/model/long.py b/model/long.py\nold mode 100644\nnew mode 100755\n", "unsupported_diff_feature"),
    ("Binary files a/model/w.bin and b/model/w.bin differ\n", "unsupported_diff_feature"),
    ("here is my diff:\n--- a/model/long.py\n+++ b/model/long.py\n@@ -1,1 +1,1 @@\n-V01 = 1\n+V01 = 2\n",
     "malformed_diff"),
    ("--- a/model/long.py\n@@ -1,1 +1,1 @@\n-V01 = 1\n+V01 = 2\n", "malformed_diff"),
    ("--- a/train.py\n+++ b/train.py\n@@ -8,3 +8,3 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n"
     "--- a/train.py\n+++ b/train.py\n@@ -8,3 +8,3 @@\n-LEARNING_RATE = 0.2\n+LEARNING_RATE = 0.3\n", "malformed_diff"),
    ("--- a/evaluate.py\n+++ b/evaluate.py\n@@ -1,5 +1,5 @@\n-def evaluate(preds, labels):\n+def evaluate(p, l):\n",
     "scope_violation"),
]


@pytest.mark.parametrize("text, reason", MALFORMED)
def test_malformed_diffs_stay_refused_with_their_reasons(env, tmp_path, text, reason):
    """Every case also has wrong counts.  The strict parser alone refuses on the structural fault where it comes
    first (headers, paths); where it met the wrong counts first it said ``malformed_hunk``, and with the recount the
    structural reason now shows."""
    node, base, _ = env
    raw, applied = apply_both(node, base, text, tmp_path)
    assert applied.status == "refused" and applied.provenance["refusal"] == reason
    assert raw.provenance["refusal"] in (reason, "malformed_hunk")


def test_bad_header_after_a_recounted_hunk_is_still_malformed_hunk(env, tmp_path):
    node, base, _ = env
    text = long_hunk(18, 25) + "@@ -x +y @@\n-V40 = 40\n+V40 = 4\n"
    rec = recount_hunk_headers(text)
    assert rec.left == "bad_hunk_header" and rec.changed
    _, applied = apply_both(node, base, text, tmp_path)
    assert applied.provenance["refusal"] == "malformed_hunk"


def test_empty_body_and_bodies_without_old_lines_are_left_to_the_strict_parser(env, tmp_path):
    node, base, _ = env
    empty = "--- a/model/long.py\n+++ b/model/long.py\n@@ -1,3 +1,3 @@\n@@ -5,1 +5,1 @@\n-V05 = 5\n+V05 = 6\n"
    rec = recount_hunk_headers(empty)
    assert rec.left == "empty_body" and rec.text == empty
    assert apply_both(node, base, empty, tmp_path, "e")[1].provenance["refusal"] == "malformed_hunk"
    # an addition without context under a header with old lines: its position would change, so it is not recounted
    noold = "--- a/model/long.py\n+++ b/model/long.py\n@@ -5,2 +5,4 @@\n+X = 1\n"
    rec = recount_hunk_headers(noold)
    assert rec.left == "no_old_lines" and rec.text == noold
    assert apply_both(node, base, noold, tmp_path, "n")[1].provenance["refusal"] == "malformed_hunk"
    # an old count of 0 in the header stays 0: the insertion point is unchanged, only the new count is fixed
    ins = "--- a/model/long.py\n+++ b/model/long.py\n@@ -5,0 +6,9 @@\n+X = 1\n+Y = 2\n"
    assert recount_hunk_headers(ins).text.splitlines()[2] == "@@ -5,0 +6,2 @@"
    right = ins.replace("+6,9", "+6,2")
    assert same_application(apply_both(node, base, ins, tmp_path, "i")[1],
                            apply_diff(copy(node, tmp_path, "ig"), right, base, write=False))


def test_a_zero_zero_hunk_with_a_correct_header_is_left_as_before(env):
    """The strict parser accepts ``@@ -0,0 +0,0 @@`` with no body as a no-op hunk; the header is right, so the
    recount leaves it (behaviour unchanged, not widened)."""
    text = "--- a/model/long.py\n+++ b/model/long.py\n@@ -0,0 +0,0 @@\n@@ -5,1 +5,1 @@\n-V05 = 5\n+V05 = 6\n"
    assert strict_ok(text) and recount_hunk_headers(text).text == text


# ----------------------------------------------------------------------------------------- the -- / ++ cases
def sql_diff(old, new, extra=""):
    return ("--- a/model/q.sql\n+++ b/model/q.sql\n"
            f"@@ -1,{old} +1,{new} @@\n -- header comment\n SELECT 1;\n--- a/q.sql\n+++ b/q.sql\n SELECT 2;\n" + extra)


def test_dash_dash_lines_with_right_counts_are_body_as_before(env, tmp_path):
    """Removing ``-- a/q.sql`` and adding ``++ b/q.sql`` writes ``--- a/q.sql`` / ``+++ b/q.sql``: with the right
    counts the header decides (as the strict parser did) and the text is unchanged."""
    node, base, _ = env
    text = sql_diff(4, 4)
    assert strict_ok(text) and recount_hunk_headers(text).text == text
    raw, applied = apply_both(node, base, text, tmp_path)
    assert raw.status == applied.status == "applied"


def test_dash_dash_pair_inside_a_recounted_plain_patch_without_a_hunk_header_after_it_is_ambiguous(env, tmp_path):
    node, base, _ = env
    text = sql_diff(9, 9)
    rec = recount_hunk_headers(text)
    assert rec.left == "ambiguous_file_header_pair" and rec.text == text
    assert apply_both(node, base, text, tmp_path)[1].status == "refused"


@pytest.mark.parametrize("pair, reason", [
    ("--- a/q.sql\n+++ b/q.sql", "scope_violation"),                 # read as a header for q.sql (not editable)
    ("--- x\n+++ y", "unsupported_diff_feature"),                      # read as a header: a rename
    ("--- a/model/q.sql\n+++ b/model/q.sql", "malformed_diff"),        # read as a header: the same file twice
])
def test_dash_dash_pair_followed_by_a_hunk_header_is_a_file_header_in_a_plain_patch(env, tmp_path, pair, reason):
    """Without ``diff --git`` a ``---``/``+++`` pair directly followed by ``@@`` ends the hunk; the strict parser then
    checks it as a file header.  Were it meant as a removed ``-- ...`` and an added ``++ ...`` line, the diff is
    refused; it is never applied as something else."""
    node, base, _ = env
    text = ("--- a/model/q.sql\n+++ b/model/q.sql\n@@ -1,9 +1,9 @@\n -- header comment\n SELECT 1;\n"
            + pair + "\n@@ -5,1 +5,1 @@\n-++ not python\n+-- python\n")
    rec = recount_hunk_headers(text)
    assert rec.left is None and rec.changed and rec.recounted[0]["body"] == [2, 2]
    _, applied = apply_both(node, base, text, tmp_path)
    assert applied.status == "refused" and applied.provenance["refusal"] == reason


def test_dash_dash_pair_is_body_in_a_git_style_patch(env, tmp_path):
    """With a ``diff --git`` line the pair is body, as in git's own recount (it ends a body only at ``@@ `` or
    ``diff ``)."""
    node, base, _ = env
    right = "diff --git a/model/q.sql b/model/q.sql\n" + sql_diff(4, 4)
    wrong = "diff --git a/model/q.sql b/model/q.sql\n" + sql_diff(9, 2)
    assert recount_hunk_headers(wrong).text == right
    good = apply_diff(copy(node, tmp_path, "good"), right, base, write=False)
    _, applied = apply_both(node, base, wrong, tmp_path)
    assert good.status == "applied" and same_application(applied, good)


def test_a_second_file_without_its_diff_git_line_is_refused_not_misapplied(env, tmp_path):
    node, base, _ = env
    text = ("diff --git a/model/long.py b/model/long.py\n--- a/model/long.py\n+++ b/model/long.py\n"
            "@@ -1,9 +1,9 @@\n V01 = 1\n-V02 = 2\n+V02 = 20\n"
            "--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n")
    _, applied = apply_both(node, base, text, tmp_path)
    assert applied.status == "refused" and applied.provenance["refusal"] in ("context_mismatch", "malformed_hunk")


# ----------------------------------------------------------------------------------------- hunks, files, markers
def test_several_hunks_and_files_with_wrong_counts(env, tmp_path):
    node, base, _ = env
    right = ("--- a/model/long.py\n+++ b/model/long.py\n"
             "@@ -2,3 +2,3 @@\n V02 = 2\n-V03 = 3\n+V03 = 30\n V04 = 4\n"
             "@@ -40,2 +40,3 @@\n V40 = 40\n+V40b = 41\n V41 = 41\n"
             "\n"
             "--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n"
             "--- a/model/net.py\n+++ b/model/net.py\n@@ -1,3 +1,3 @@\n-WIDTH = 64\n+WIDTH = 128\n"
             " # width of the hidden layer\n DEPTH = 2\n")
    wrong = (right.replace("@@ -2,3 +2,3 @@", "@@ -2,7 +2,1 @@").replace("@@ -40,2 +40,3 @@", "@@ -40 +40 @@")
             .replace("@@ -8,1 +8,1 @@", "@@ -8,4 +8,4 @@").replace("@@ -1,3 +1,3 @@", "@@ -1,2 +1,2 @@"))
    assert strict_ok(right) and not strict_ok(wrong)
    rec = recount_hunk_headers(wrong)
    assert rec.text == right and len(rec.recounted) == 4 and rec.hunks == 4
    good = apply_diff(copy(node, tmp_path, "good"), right, base, write=False)
    _, applied = apply_both(node, base, wrong, tmp_path)
    assert good.status == "applied" and same_application(applied, good)
    assert normalised_diff_sha256(parse_unified_diff(rec.text)) == good.provenance["normalised_diff_sha256"]


def test_blank_lines_before_the_next_file_or_the_end_are_separators_and_before_a_hunk_are_context(env):
    text = ("--- a/model/long.py\n+++ b/model/long.py\n@@ -2,9 +2,9 @@\n V02 = 2\n-V03 = 3\n+V03 = 30\n\n\n"
            "--- a/train.py\n+++ b/train.py\n@@ -8,9 +8,9 @@\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n\n")
    out = recount_hunk_headers(text).text.splitlines()
    assert out[2] == "@@ -2,2 +2,2 @@" and out[10] == "@@ -8,1 +8,1 @@"
    # a line of one space is the canonical empty context line: kept as context
    assert recount_hunk_headers(text.replace("+V03 = 30\n\n\n", "+V03 = 30\n \n")).text.splitlines()[2] \
        == "@@ -2,3 +2,3 @@"
    # blank lines before a hunk header are empty context, as git counts them
    inner = "--- a/model/long.py\n+++ b/model/long.py\n@@ -2,9 +2,9 @@\n V02 = 2\n\n@@ -9 +9 @@\n-V09 = 9\n+V09 = 0\n"
    assert recount_hunk_headers(inner).text.splitlines()[2] == "@@ -2,2 +2,2 @@"


@pytest.mark.parametrize("right", [
    # the last line changes and the file keeps no final newline
    "--- a/model/tail.txt\n+++ b/model/tail.txt\n@@ -2,2 +2,2 @@\n beta\n-gamma\n\\ No newline at end of file\n"
    "+delta\n\\ No newline at end of file\n",
    # the edit adds the final newline
    "--- a/model/tail.txt\n+++ b/model/tail.txt\n@@ -2,2 +2,2 @@\n beta\n-gamma\n\\ No newline at end of file\n"
    "+gamma\n",
    # an edit before an unchanged last line without a newline
    "--- a/model/tail.txt\n+++ b/model/tail.txt\n@@ -1,3 +1,3 @@\n-alpha\n+ALPHA\n beta\n gamma\n"
    "\\ No newline at end of file\n",
])
def test_no_newline_markers_are_not_counted(env, tmp_path, right):
    node, base, _ = env
    wrong = right.replace("@@ -2,2 +2,2 @@", "@@ -2,5 +2,6 @@").replace("@@ -1,3 +1,3 @@", "@@ -1,4 +1,4 @@")
    assert strict_ok(right) and not strict_ok(wrong)
    assert recount_hunk_headers(wrong).text == right
    good = apply_diff(copy(node, tmp_path, "good"), right, base, write=False)
    _, applied = apply_both(node, base, wrong, tmp_path)
    assert good.status in ("applied", "inert") and same_application(applied, good)   # a final newline alone: inert
    assert apply_diff(copy(node, tmp_path, "w"), recount_hunk_headers(wrong).text, base).provenance["refusal"] is None


# ----------------------------------------------------------------------------------------- identity on accepted diffs
def _random_diff(rng, lines, path, blank=" "):
    """A correctly counted diff of 1-3 hunks over *lines* (context written as ' x', blank context as *blank*)."""
    picks = sorted(rng.sample(range(2, len(lines) - 6), rng.randint(1, 3)))
    starts, last_end = [], -1
    for p in picks:
        if p > last_end + 1:
            starts.append(p)
            last_end = p + 5
    out = [f"--- a/{path}", f"+++ b/{path}"]
    for s in starts:
        pre, cut, post = rng.randint(0, 2), rng.randint(0, 2), rng.randint(0, 2)
        adds = [f"NEW_{s}_{k} = {k}" for k in range(rng.randint(0 if cut else 1, 3))]
        old = lines[s - pre:s + cut + post]
        body = [(" " + x if x else blank) for x in lines[s - pre:s]] + ["-" + x for x in lines[s:s + cut]]
        body += ["+" + x for x in adds] + [(" " + x if x else blank) for x in lines[s + cut:s + cut + post]]
        if not old:
            continue
        out.append(f"@@ -{s - pre + 1},{len(old)} +{s - pre + 1},{pre + len(adds) + post} @@")
        out += body
    return "\n".join(out) + "\n" if len(out) > 2 else None


def test_every_diff_the_strict_parser_accepts_is_returned_byte_identical():
    rng = random.Random(20261124)
    lines = LONG[:30] + [""] + LONG[30:]
    corpus = [long_hunk(17, 21), sql_diff(4, 4), "--- /dev/null\n+++ b/model/new.py\n@@ -0,0 +1,2 @@\n+A = 1\n+B = 2\n",
              "--- a/model/net.py\n+++ /dev/null\n@@ -1,3 +0,0 @@\n-WIDTH = 64\n-# width of the hidden layer\n"
              "-DEPTH = 2\n"]
    for _ in range(400):
        d = _random_diff(rng, lines, "model/long.py", blank=rng.choice([" ", ""]))
        if d:
            corpus.append(d + rng.choice(["", "\n", "\n\n"]))
    accepted = [d for d in corpus if strict_ok(d)]
    assert len(accepted) > 300
    for d in accepted:
        rec = recount_hunk_headers(d)
        assert rec.text == d and not rec.changed, d


def test_perturbed_counts_recount_back_to_the_correct_text():
    rng = random.Random(7)
    checked = 0
    for _ in range(400):
        d = _random_diff(rng, LONG, "model/long.py")
        if not d:
            continue
        lines = d.split("\n")
        for k, line in enumerate(lines):
            if line.startswith("@@"):
                head = line.split(" @@")[0][3:].split(" ")
                o, nw = head[0][1:].split(","), head[1][1:].split(",")
                lines[k] = f"@@ -{o[0]},{rng.randint(0, 40)} +{nw[0]},{rng.randint(0, 40)} @@"
        wrong = "\n".join(lines)
        if wrong == d:
            continue
        got = recount_hunk_headers(wrong)
        if got.left == "no_old_lines":          # a context-free insertion with a header that had old lines
            continue
        assert got.text == d, (wrong, got.text)
        checked += 1
    assert checked > 300


def test_not_text_is_returned_as_is():
    assert recount_hunk_headers(None).text is None and not recount_hunk_headers(None).changed


# ----------------------------------------------------------------------------------------- campaign and audit
def reply(diff, summary="s"):
    return json.dumps({"rationale": "r", "summary": summary, "objective": "o", "diff": diff})


class ScriptedChat:
    def __init__(self, replies):
        self.replies = list(replies)

    def __call__(self, payload):
        return self.replies.pop(0), {"prompt_tokens": 10, "completion_tokens": 5}


LR_WRONG = "--- a/train.py\n+++ b/train.py\n@@ -7,4 +7,6 @@\n MODE = \"ok\"\n-LEARNING_RATE = 0.1\n+LEARNING_RATE = 0.2\n"
LR_RIGHT = LR_WRONG.replace("@@ -7,4 +7,6 @@", "@@ -7,2 +7,2 @@")
LR_NEXT = "--- a/train.py\n+++ b/train.py\n@@ -8,1 +8,1 @@\n-LEARNING_RATE = 0.2\n+LEARNING_RATE = 0.3\n"


def test_campaign_applies_the_recounted_copy_and_keeps_the_managers_text(env, tmp_path):
    node, base, spec = env
    chat = ScriptedChat([reply(LR_WRONG, "lr up"), reply(LR_NEXT, "lr up again")])
    entries = run_v2_diff_campaign(spec, node, tmp_path / "runs", V2DiffManager(chat, "fixture", seed=1), LIMITS, 2,
                                   enable_v2_diff=True, unisolated_reason="test fixture (not isolated)")
    assert [e["decision"] for e in entries] == ["baseline", "kept", "kept"]
    step1, step2 = entries[1], entries[2]
    assert step1["proposal"]["diff"] == LR_WRONG                     # O1: the manager's text as written
    block = step1["diff_recount"]
    assert block["rule"] == DIFF_RECOUNT_RULE and block["changed"] and block["hunks_recounted"] == 1
    assert block["recounted"] == [{"line": 3, "header": [4, 6], "body": [2, 2]}]
    assert block["raw_diff_sha256"] == hashlib.sha256(LR_WRONG.encode()).hexdigest()
    assert block["applied_diff_sha256"] == hashlib.sha256(LR_RIGHT.encode()).hexdigest()
    run = step1["runs"][0]
    assert run["diff"]["diff_text"] == LR_RIGHT and run["diff"]["diff_sha256"] == block["applied_diff_sha256"]
    # the candidate's code state is that of the correctly counted diff
    assert run["code_state_sha256"] == apply_diff(copy(node, tmp_path, "c"), LR_RIGHT, base,
                                                  write=False).provenance["code_state_sha256"]
    # nothing recounted: the run applies the manager's text itself
    assert step2["diff_recount"]["changed"] is False and step2["runs"][0]["diff"]["diff_text"] == LR_NEXT
    assert step2["diff_recount"]["raw_diff_sha256"] == step2["diff_recount"]["applied_diff_sha256"]
    ledger = [json.loads(x) for x in (tmp_path / "runs" / "campaign_ledger.jsonl").read_text().splitlines()]
    assert ledger[1]["proposal"]["diff"] == LR_WRONG and ledger[1]["diff_recount"] == block


def test_campaign_records_a_refusal_the_recount_leaves_to_the_strict_parser(env, tmp_path):
    node, base, spec = env
    bad = "--- a/train.py\n+++ b/train.py\n@@ -8,3 +8,3 @@\n-LEARNING_RATE = 0.1\n*LEARNING_RATE = 0.2\n"
    entries = run_v2_diff_campaign(spec, node, tmp_path / "runs", V2DiffManager(ScriptedChat([reply(bad)]), "f"),
                                   LIMITS, 1, enable_v2_diff=True, unisolated_reason="test fixture (not isolated)")
    assert entries[1]["decision"] == "not_run_refused"
    assert entries[1]["runs"][0]["diff"]["refusal"] == "malformed_hunk"
    assert entries[1]["diff_recount"]["left_to_strict_parser"] == "bad_line_tag"


def test_audit_rematerialises_the_managers_texts_to_the_campaigns_states(env, tmp_path):
    node, base, spec = env
    chat = ScriptedChat([reply(LR_WRONG), reply(LR_NEXT.replace("@@ -8,1 +8,1 @@", "@@ -8,5 +8,5 @@"))])
    entries = run_v2_diff_campaign(spec, node, tmp_path / "runs", V2DiffManager(chat, "fixture", seed=1), LIMITS, 2,
                                   enable_v2_diff=True, unisolated_reason="test fixture (not isolated)")
    assert [e["decision"] for e in entries] == ["baseline", "kept", "kept"]
    diffs = [e["proposal"]["diff"] for e in entries[1:]]        # what the paper's audit_plan reads (proposal.diff)
    assert all(e["diff_recount"]["changed"] for e in entries[1:])
    units = [{"unit": "u1", "code_state_sha256": entries[1]["runs"][0]["code_state_sha256"], "diffs": diffs[:1]},
             {"unit": "u2", "code_state_sha256": entries[2]["runs"][0]["code_state_sha256"], "diffs": diffs}]
    root = tmp_path / "states"
    root.mkdir()
    states = audit_g5.prepare_states(spec, node, {"units": units, "interpreter": "python3"}, root)
    assert states["u2"][1] == recount_hunk_headers(diffs[1]).text     # the runner gets the campaign's text
    # without the recount the chain would not re-materialise (the G.4 rule refuses the manager's text)
    assert apply_diff(copy(node, tmp_path, "s"), diffs[0], base, write=False).provenance["refusal"] == "malformed_hunk"
    inc = states["u2"][0]
    assert code_state_hashes(inc, base)[0] == entries[1]["runs"][0]["code_state_sha256"]


def test_audit_chunk_record_names_the_rule(env, tmp_path, monkeypatch):
    """The chunk record carries ``diff_rule`` and ``diffs_recounted`` (written before the states are built)."""
    node, base, spec = env
    seen = {}

    def fake_prepare(spec_, node_dir, data, root):
        raise RuntimeError("stop here")

    monkeypatch.setattr(audit_g5, "prepare_states", fake_prepare)
    monkeypatch.setattr(audit_g5, "load_node", lambda data, root: (spec, node, {}))
    monkeypatch.setattr(audit_g5, "build_limits", lambda *a: LIMITS)
    data = {"node": "fake_task", "run_id": "r", "wave": 1, "chunk": 0, "K": 10, "runs": [], "cgroup_parent": "/nope",
            "units": [{"unit": "u", "code_state_sha256": "0" * 64, "diffs": [LR_WRONG, LR_NEXT]}]}
    rec = audit_g5.run_audit(data, tmp_path, tmp_path / "out", gpu_uuids=(), cpu_list=(0,))
    seen.update(rec)
    assert seen["diff_rule"] == DIFF_RECOUNT_RULE and seen["diffs_recounted"] == 1 and "stop here" in seen["stop"]


def test_diff_support_itself_is_unchanged_and_still_strict():
    """diff_support (P5 measurement path; also noise_floor's declared edits) still refuses wrong counts."""
    with pytest.raises(DiffRefused, match="malformed_hunk"):
        parse_unified_diff(LR_WRONG)
