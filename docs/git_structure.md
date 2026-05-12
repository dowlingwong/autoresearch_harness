# Git Structure

This repository is the superproject for the research harness. It should record
the harness code, node definitions, tests, paper-facing source, planning files,
and pinned versions of small upstream-backed repositories. It should not record
generated experiment logs, transient ledgers, node-local runtime state, model
checkpoints, or local datasets.

## Intended Layout

```text
autoresearch_harness/                         # parent superproject
  src/ scripts/ configs/ tests/ nodes/ docs/   # parent-owned harness code
  plan/                                        # parent-owned planning/status
  paper/                                       # parent-owned paper source plus generated outputs
  harness/claw-code/                           # submodule, upstream-backed worker repo
  A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/
                                               # submodule, paper/Overleaf repo
```

The parent repo records submodules as commit pointers. Changes inside a
submodule must be committed in that submodule first, then the parent commits the
updated pointer. This gives the parent a reproducible snapshot while preserving
the ability to pull from each submodule's upstream.

## What The Parent Tracks

- Harness source: `src/`
- Scripts and configs: `scripts/`, `configs/`
- Node source and lightweight node docs: `nodes/`
- Tests: `tests/`
- Planning and design docs: `plan/`, `docs/`
- Intentional paper source and reviewed tables/figures: `paper/`
- Submodule pointers:
  - `harness/claw-code`
  - `A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation`

## What The Parent Does Not Track

- `experiments/**` except directory `.gitkeep` files
- node runtime state such as `.autoresearch_state.json`, `results.tsv`,
  `experiment_memory.jsonl`, `run.log`, and `artifacts/`
- checkpoints such as `*.pt`, `*.pth`, `*.ckpt`
- local datasets such as `*.h5` and `*.hdf5`
- LaTeX build byproducts

## Normal Workflows

Clone with submodules:

```bash
git clone --recurse-submodules <parent-url>
```

Initialize submodules after a plain clone:

```bash
git submodule update --init --recursive
```

Update `claw-code` from its upstream:

```bash
git -C harness/claw-code fetch upstream
git -C harness/claw-code checkout main
git -C harness/claw-code merge upstream/main
git -C harness/claw-code push origin main
git add harness/claw-code
git commit -m "chore: update claw-code submodule"
```

Commit local changes inside a submodule:

```bash
git -C harness/claw-code status
git -C harness/claw-code add <files>
git -C harness/claw-code commit -m "<message>"
git -C harness/claw-code push origin main
git add harness/claw-code
git commit -m "chore: pin claw-code submodule"
```

Check parent and submodule state:

```bash
git status --short --branch
git submodule status --recursive
git diff --submodule=log
```

## Recommended Next Cleanup

The current paper submodule name is long and root-level. A cleaner future
layout would move it to `paper/manuscript` or replace it with the active
`paper/kdd_aae_2026` source, but that should be done as a separate migration
after deciding which paper repository is canonical.
