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
    make, served, violations, open_validation = ar_budget.budgeted(loader, 3 * 2 * 4)
    it = make(None, 2, 4, "train")
    for _ in range(3):
        next(it)
    assert served() == 24
    with pytest.raises(ar_budget.BudgetExhausted):
        next(it)
    with pytest.raises(ar_budget.ValidationRequested):
        make(None, 2, 4, "val")
    assert "val" not in calls                    # refused before the real loader is built
    assert violations() == ["budget", "validation"]  # recorded even if train.py catches the exceptions
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
    "caught_validation": (PROBE_HEAD + "try:\n    make_dataloader(tokenizer, 1, MAX_SEQ_LEN, 'val')\n"
                          "except BaseException:\n    pass\nnext(loader); next(loader)\n", 8, "caught the refusal"),
}

MODEL_HEAD = PROBE_HEAD + """next(loader); next(loader)
import torch
class M(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.e = torch.nn.Embedding(vocab, vocab)
    def forward(self, x, targets=None, reduction="mean"):
        logits = self.e(SHIFT(x)).float()
        if targets is None:
            return logits
        return torch.zeros_like(targets, dtype=torch.float32)   # an edited loss: zero (the metric must ignore it)
model = M(tokenizer.get_vocab_size())
"""
EVALUATED = {   # train.py stand-ins that use the whole budget and leave a model; (exit, check of stdout)
    "causal_with_forgeries": (MODEL_HEAD.replace("SHIFT(x)", "x") + """import atexit, sys
atexit.register(lambda: print("AR_METRIC val_bpb=0.1", flush=True))   # a line after the wrapper's
real = sys.stdout
class Filter:
    def write(self, text):
        return real.write(text.replace("AR_METRIC val_bpb=", "AR_METRIC val_bpb=0.0001 # "))
    def flush(self):
        real.flush()
sys.stdout = Filter()                                                  # a rewrite of the wrapper's line
""", 0, None),
    "sets_the_default_device": (MODEL_HEAD.replace("SHIFT(x)", "x") + "torch.set_default_device('cpu')  # an ordinary edit\n",
                                0, None),
    "peeks_at_the_next_token": (MODEL_HEAD.replace("SHIFT(x)", "torch.roll(x, -1, dims=1)"), 9, "not causal"),
    "peeks_with_an_infinite_logit": (MODEL_HEAD.replace("SHIFT(x)", "torch.roll(x, -1, dims=1)").replace(
        "        logits = self.e(", "        logits = self.e(").replace(
        "        if targets is None:", "        logits = logits.clone(); logits[..., -1] = float('-inf')\n        if targets is None:"),
        9, "not causal"),
    "updates_weights_in_forward": (MODEL_HEAD.replace("SHIFT(x)", "x").replace(
        "        if targets is None:", "        with torch.no_grad():\n            self.e.weight.add_(1e-3)\n        if targets is None:"),
        10, "weights or buffers changed"),
    "returns_a_tensor_subclass": (MODEL_HEAD.replace("SHIFT(x)", "x").replace(
        "model = M(tokenizer.get_vocab_size())", """class Liar(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs or {})
_forward = M.forward
M.forward = lambda self, x, targets=None, reduction="mean": _forward(self, x, targets, reduction).as_subclass(Liar)
model = M(tokenizer.get_vocab_size())"""), 4, "plain torch.Tensor"),
    "leaves_a_dispatch_mode_active": (MODEL_HEAD.replace("SHIFT(x)", "x") + """from torch.utils._python_dispatch import TorchDispatchMode
class Zero(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))
Zero().__enter__()
""", 4, "dispatch mode is active"),
    "hides_tensors_from_the_fingerprint": (MODEL_HEAD.replace("SHIFT(x)", "x").replace(
        "        if targets is None:", "        with torch.no_grad():\n            self.e.weight.add_(1e-3)\n        if targets is None:").replace(
        "model = M(tokenizer.get_vocab_size())", """M._named_members = lambda self, *a, **k: iter(())
M.named_modules = lambda self, *a, **k: iter(())
model = M(tokenizer.get_vocab_size())"""), 10, "weights or buffers changed"),
    "plants_a_data_file": (MODEL_HEAD.replace("SHIFT(x)", "x") + """import os, pathlib
d = pathlib.Path("data").resolve() / "data"
os.chmod(d, 0o755)
(d / "shard_00010.parquet").write_bytes(b"planted")
""", 5, "train.py changed the data"),
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


@pytest.mark.parametrize("probe", sorted(EVALUATED))
def test_wrapper_metric_is_frozen(tmp_path, monkeypatch, probe):
    """Review round 2, findings 1 and 3: the metric's cross-entropy is computed by the wrapper from the logits (an
    edited zero loss does not reach it), a model that reads the next token is refused (exit 9), and the wrapper's
    result line can be neither rewritten through sys.stdout nor followed by an atexit line."""
    if not ps.have("torch", "pyarrow", "tiktoken", "rustbpe"):
        pytest.skip("needs torch, pyarrow, tiktoken and rustbpe in the test interpreter")
    work = _prepare_smoke_copy(tmp_path, monkeypatch)
    code, want, message = EVALUATED[probe]
    (work / "train.py").write_text(code)
    env = dict(os.environ, AR_TRAIN_SEED="8001", AR_SPLIT_SEED="8001")
    res = subprocess.run([sys.executable, "run_fixed_budget.py", "--tokens", "512"], cwd=work, env=env,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == want, (res.returncode, res.stdout[-2000:], res.stderr[-2000:])
    metric_lines = [line for line in res.stdout.splitlines() if line.startswith("AR_METRIC")]
    if message is not None:
        assert message in res.stdout and not metric_lines
        return
    assert len(metric_lines) == 1 and res.stdout.rstrip().splitlines()[-1] == metric_lines[0], res.stdout[-1000:]
    value = float(metric_lines[0].split("=", 1)[1])
    assert value > 1.0   # an untrained embedding: far from the forged 0.1 / 0.0001 and from the zero edited loss


def test_unlisted_data_file_refused(tmp_path):
    """Review round 2, finding 2: a file the manifest does not list (a planted shard) stops the run (exit 5)."""
    ar_data = ps.load_node_module(NODE, "ar_data")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "shard_00000.parquet").write_bytes(b"a")
    ar_data.write_manifest(tmp_path, NODE, ["data/shard_00000.parquet"], {"test": True})
    assert ar_data.verify(tmp_path, NODE, ["data/shard_00000.parquet"])["node"] == NODE
    (tmp_path / "data" / "shard_00010.parquet").write_bytes(b"planted")
    with pytest.raises(SystemExit) as stop:
        ar_data.verify(tmp_path, NODE, ["data/shard_00000.parquet"])
    assert stop.value.code == ar_data.EXIT_DATA


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


@pytest.mark.parametrize("node", [NODE])
def test_output_seal(node):
    """Review findings (partial line before seal, thread writes): the result line still comes last."""
    ps.check_output_seal(node)
