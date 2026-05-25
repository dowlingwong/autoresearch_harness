# Environment Setup

This project requires **Python 3.11+** (`StrEnum` is stdlib in 3.11; it is not available in 3.9/3.10 without the optional `strenum` backport).  
The `.python-version` file at the repo root pins the interpreter to 3.11 for `uv`.

---

## Recommended: uv (fast, reproducible)

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the repo root — creates .venv with Python 3.11 and installs all deps
uv sync

# Run any script through uv so it picks up the pinned interpreter + venv
uv run python3 scripts/run_campaign.py --help
uv run python3 scripts/run_counterfactual.py --help
uv run python3 scripts/analyze_counterfactual.py --help
uv run pytest -p no:cacheprovider
```

`uv sync` reads `pyproject.toml` and the `.python-version` pin.  
The resulting `.venv/` is local to the repo and is excluded from git.

---

## Fallback: pip + system Python 3.11

If you have a system or conda Python 3.11 available:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python3 scripts/run_counterfactual.py --help
```

---

## Why not Anaconda / conda Python 3.9?

The default macOS Anaconda environment ships Python 3.9.  Running scripts
directly with that interpreter raises:

```
ModuleNotFoundError: No module named 'strenum'
```

`strenum` is an optional backport package, not installed by default.  
Use `uv run` (which enforces the `.python-version` pin) instead of invoking
`python3` from the base conda environment.

---

## deepthought2 (NVIDIA GPU cluster)

See `scripts/sync_to_deepthought2.sh` for how to push the repo to
`/ceph/dwong/autoresearch_harness/` on deepthought2.

On deepthought2, run the same `uv sync` setup above (uv is available in the
cluster's module system, or install it via the curl one-liner).

For GPU jobs (ClawWorker / autoresearch_linux node):

```bash
# On deepthought2 after syncing
cd /ceph/dwong/autoresearch_harness
uv sync
uv run python3 scripts/run_counterfactual.py \
    --node autoresearch_linux \
    --base kdd_cf_arlinux \
    --budget 30 \
    --node-root /ceph/dwong/autoresearch_harness/nodes/autoresearch_linux \
    --model deepseek/deepseek-v4-flash \
    --host http://localhost:11434 \
    --use-claw-worker
```

---

## Key dependencies

| Package | Role |
|---|---|
| `numpy` | Bootstrap CI in `analyze_counterfactual.py` |
| `langgraph` | LangGraph manager backend |
| `strenum` | StrEnum backport (Python < 3.11 only; not needed with uv) |
| `pytest` | Test suite |
| `pyyaml` | NodeSpec YAML loading |

Full dependency list: see `pyproject.toml`.
