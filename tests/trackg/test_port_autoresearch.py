"""Port kit U3, suite autoresearch: node autoresearch_fixedtokens.

Static checks always run.  The synthetic smoke needs torch, pyarrow, tiktoken
and rustbpe in the interpreter running the tests.  It writes two small
parquet shards of random words (no climbmix data), trains the tokenizer on
them with the node's own ``prepare.train_tokenizer``, and runs with a reduced
token budget.  For CPU speed, the smoke's copy of the node (never the
repository node) gets two test-only changes in its frozen prepare.py:
MAX_SEQ_LEN 2048 -> 256 and EVAL_TOKENS -> 16,384.  The smoke also runs with
TORCHDYNAMO_DISABLE=1 (torch.compile becomes a no-op): on the port's CPU test
host, torch 2.6's CPU inductor build of autoresearch's compiled Muon step
crashed the process (segmentation fault in ``_step_muon``), unrelated to the
port.  The compiled GPU path is exercised only by the census on the GPU host.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from . import port_support as ps

NODE = "autoresearch_fixedtokens"
NODES = [NODE]
FILES = [*[f"data/shard_{i:05d}.parquet" for i in range(10)], "data/shard_06542.parquet",
         "tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt"]
BROKEN = ("            loss = model(x, y)\n", '            loss = model(x, y) * float("nan")\n')


def test_static():
    ps.check_static(NODE, NODES)


def test_manifest_complete():
    ps.check_manifest(NODE)


def test_frozen_path_edit_refused():
    ps.check_scope(NODE)


def test_prepare_is_the_harness_copy():
    node = ps.node_dir(NODE)
    source = ps.ROOT / "nodes" / "autoresearch-macos" / "prepare.py"
    assert (node / "prepare.py").read_bytes() == source.read_bytes()


def test_token_budget_is_whole_steps():
    spec = ps.load(NODE)
    assert spec.run_budget.kind == "tokens" and spec.run_budget.value % 2**16 == 0
    assert "{budget}" in spec.run_command and "--tokens" in spec.run_command


def synthetic_corpus(d: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.RandomState(0)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    words = ["".join(rng.choice(letters, rng.randint(2, 9))) for _ in range(3000)]
    docs = [" ".join(rng.choice(words, rng.randint(50, 400))) + "." for _ in range(2500)]
    (d / "data").mkdir()
    for i in range(10):   # the wrapper requires all ten training shards and the validation shard
        pq.write_table(pa.table({"text": docs[i * 200:(i + 1) * 200]}), d / "data" / f"shard_{i:05d}.parquet")
    pq.write_table(pa.table({"text": docs[2000:]}), d / "data" / "shard_06542.parquet")
    env = dict(os.environ, AUTORESEARCH_CACHE_DIR=str(d))
    subprocess.run([sys.executable, "-c", "import prepare; prepare.train_tokenizer()"], cwd=ps.node_dir(NODE),
                   env=env, check=True, capture_output=True)
    return {"shapes": "random-word parquet shards; tokenizer trained on them by prepare.train_tokenizer"}


SMOKE = {NODE: {"broken": BROKEN, "make": synthetic_corpus, "divergent": set(), "torch_node": True}}


def test_budget_closure_counts_refuses_and_guards_validation():
    """Review round 1, findings 1 and 8 (no torch needed): the counter is a closure, a request beyond the budget and
    a validation request while train.py runs raise BaseExceptions, and validation opens only when the wrapper says."""
    ar_budget = ps.load_node_module(NODE, "ar_budget")
    calls = []

    def loader(tokenizer, B, T, split, *a, **k):
        calls.append(split)
        return iter([("x", "y")] * 10)
    make, served, open_validation = ar_budget.budgeted(loader, 3 * 2 * 4)
    it = make(None, 2, 4, "train")
    for _ in range(3):
        next(it)
    assert served() == 24
    with pytest.raises(ar_budget.BudgetExhausted):
        next(it)
    with pytest.raises(ar_budget.ValidationRequested):
        make(None, 2, 4, "val")
    assert "val" not in calls                    # refused before the real loader is built
    open_validation()
    assert next(make(None, 2, 4, "val")) == ("x", "y")
    assert not hasattr(ar_budget, "_served")     # no writable module counter
    assert issubclass(ar_budget.BudgetExhausted, BaseException) and \
        not issubclass(ar_budget.ValidationRequested, Exception)


def _prepare_smoke_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")  # see the module docstring
    work = ps.smoke_copy(NODE, tmp_path, synthetic_corpus, FILES)
    prepare = work / "prepare.py"
    text = prepare.read_text()
    for old, new in (("MAX_SEQ_LEN = 2048 ", "MAX_SEQ_LEN = 256 "),
                     ("EVAL_TOKENS = 40 * 524288 ", "EVAL_TOKENS = 4 * 16 * 256 ")):
        assert text.count(old) == 1
        text = text.replace(old, new)
    prepare.write_text(text)  # test-only change of the smoke copy (see the module docstring)
    return work


PROBE_HEAD = """from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader
tokenizer = Tokenizer.from_directory()
loader = make_dataloader(tokenizer, 1, MAX_SEQ_LEN, "train")
"""
PROBES = {   # train.py stand-ins (budget 2 x 256 tokens) and the exit the wrapper must give
    "under_budget": (PROBE_HEAD + "next(loader)\n", 3, "used 256 of 512"),
    "over_request": (PROBE_HEAD + "next(loader); next(loader); next(loader)\n", 7, "more than the budget"),
    "validation": (PROBE_HEAD + "make_dataloader(tokenizer, 1, MAX_SEQ_LEN, 'val')\n", 8, "validation split"),
    "forged_exit": (PROBE_HEAD + "next(loader); next(loader)\nprint('AR_METRIC val_bpb=0.1')\n"
                    "import sys; sys.exit(0)\n", 4, "SystemExit"),
}


@pytest.mark.parametrize("probe", sorted(PROBES))
def test_wrapper_budget_exits(tmp_path, monkeypatch, probe):
    """Review round 1, findings 1, 8 and 12: exit 3 (budget not used), 7 (over-request), 8 (validation requested
    while train.py runs), 4 (SystemExit after a forged metric line: the budget check is not skipped)."""
    if not ps.have("torch", "pyarrow", "tiktoken", "rustbpe"):
        pytest.skip("needs torch, pyarrow, tiktoken and rustbpe in the test interpreter")
    work = _prepare_smoke_copy(tmp_path, monkeypatch)
    code, want, message = PROBES[probe]
    (work / "train.py").write_text(code)
    env = dict(os.environ, AR_TRAIN_SEED="8001", AR_SPLIT_SEED="8001")
    res = subprocess.run([sys.executable, "run_fixed_budget.py", "--tokens", "512"], cwd=work, env=env,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == want, (res.returncode, res.stdout[-2000:], res.stderr[-2000:])
    assert message in res.stdout
    assert "AR_TOKENS consumed" not in res.stdout


def test_synthetic_smoke(tmp_path, monkeypatch):
    if not ps.have("torch", "pyarrow", "tiktoken", "rustbpe"):
        pytest.skip("needs torch, pyarrow, tiktoken and rustbpe in the test interpreter")
    work = _prepare_smoke_copy(tmp_path, monkeypatch)
    broken = ps.broken_diff(NODE, *BROKEN)
    records, per_edit = ps.run_smoke(NODE, tmp_path, work, 2**16, broken)
    ps.check_smoke(records, per_edit, set())


def test_prepare_data_requires_the_exact_shard_set_and_a_data_root(tmp_path, monkeypatch):
    """Review round 1, findings 2 and 10: no setup outside $AR_TRACKG_DATA_ROOT, no reuse of a directory that
    already holds parquet files, and the required list names all ten training shards plus the validation shard."""
    monkeypatch.syspath_prepend(str(ps.node_dir(NODE)))
    prepare_data = ps.load_node_module(NODE, "prepare_data")
    assert prepare_data.REQUIRED_SHARDS == sorted([f"shard_{i:05d}.parquet" for i in range(10)] +
                                                  ["shard_06542.parquet"])
    monkeypatch.delenv("AR_TRACKG_DATA_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit, match="AR_TRACKG_DATA_ROOT"):
        prepare_data.main()
    monkeypatch.setenv("AR_TRACKG_DATA_ROOT", str(tmp_path))
    (tmp_path / "reused" / "data").mkdir(parents=True)
    (tmp_path / "reused" / "data" / "shard_00042.parquet").write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--out", str(tmp_path / "reused")])
    with pytest.raises(SystemExit, match="already holds parquet"):
        prepare_data.main()
    run = ps.load_node_module(NODE, "run_fixed_budget")
    assert sorted(run.REQUIRED_FILES) == sorted(prepare_data.REQUIRED_FILES)
