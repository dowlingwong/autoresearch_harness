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
FILES = ["data/shard_00000.parquet", "data/shard_06542.parquet", "tokenizer/tokenizer.pkl",
         "tokenizer/token_bytes.pt"]


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
    pq.write_table(pa.table({"text": docs[:2000]}), d / "data" / "shard_00000.parquet")
    pq.write_table(pa.table({"text": docs[2000:]}), d / "data" / "shard_06542.parquet")
    env = dict(os.environ, AUTORESEARCH_CACHE_DIR=str(d))
    subprocess.run([sys.executable, "-c", "import prepare; prepare.train_tokenizer()"], cwd=ps.node_dir(NODE),
                   env=env, check=True, capture_output=True)
    return {"shapes": "random-word parquet shards; tokenizer trained on them by prepare.train_tokenizer"}


def test_synthetic_smoke(tmp_path, monkeypatch):
    if not ps.have("torch", "pyarrow", "tiktoken", "rustbpe"):
        pytest.skip("needs torch, pyarrow, tiktoken and rustbpe in the test interpreter")
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
    broken = ps.broken_diff(NODE, "            loss = model(x, y)\n", '            loss = model(x, y) * float("nan")\n')
    records, per_edit = ps.run_smoke(NODE, tmp_path, work, 2**16, broken)
    ps.check_smoke(records, per_edit, set())
