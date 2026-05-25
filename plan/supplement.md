# Supplement And Reproducibility Checklist

Purpose: use `supplement.tex` and the artifact bundle to make the KDD workshop
submission auditable without turning the supplement into a second paper. The
main paper must remain self-contained; the supplement should index evidence,
methods, environment details, and regeneration steps.

## 1. Supplement Positioning

- [x] State that the supplement is a reproducibility appendix, not required for
      understanding the main claims.
  - Requirement: no new central claim appears only in the supplement.
  - Include: one opening paragraph explaining that the main paper reports the
    claims and the supplement indexes evidence and regeneration details.
- [x] Keep governance-first framing.
  - Requirement: organize around auditability, provenance, lifecycle
    classification, and reproducibility before task metrics.
- [x] Preserve anonymity if required by the submission system.
  - Requirement: use an anonymized artifact URL or placeholder during review.
  - Do not include personal usernames, private host login names, API keys, or
    non-anonymized repository ownership if double-anonymous review applies.

## 2. Artifact And Code Availability

- [x] Add a code/artifact availability subsection.
  - Include: repository URL or anonymized artifact bundle URL.
  - Include: commit hash, tag, or archive checksum for the submitted state.
  - Include: license and minimal citation instruction.
- [x] Explain the two evidence locations.
  - Include: local source checkout contains code, configs, paper source,
    ledgers, summary tables, and generated figures intended for submission.
  - Include: `/ceph/dwong/autoresearch_harness/experiments/` contains execution
    server patch/log artifacts referenced by provenance IDs.
  - Requirement: clearly say `/ceph` paths are real on `deepthought2`, not
    missing local files.
- [x] Add artifact manifest guidance.
  - Include: `artifact_manifest.json` path.
  - Include: how each ledger record references proposal, patch, raw log,
    parsed metrics, decision, and provenance IDs.
  - Requirement: reviewers should be able to map a table row to the underlying
    ledger and artifact reference.

## 3. Required Supplement Content

- [x] Include OpenML detail table.
  - Current hook: `\input{tables/openml_campaign_summary}` in `supplement.tex`.
  - Requirement: table describes public node, budget, seed count, trials,
    acceptance rate, invalid rate, mean best AUC, and 95% CI.
- [x] Include campaign ID inventory.
  - Requirement: list campaign IDs by node, arm/memory mode, seed, and budget.
  - Requirement: distinguish the 1,445-trial governed evidence set from
    illustrative validation/counterfactual campaigns.
- [x] Include hardware and software environment details.
  - Required fields: `deepthought2`, NVIDIA L40S, CUDA 12.8, Ubuntu 22.04,
    PyTorch version if known, Python version, package manager, manager model,
    worker model, temperatures, and execution host artifact path.
- [x] Include bootstrap CI methodology.
  - Required fields: 10,000 resamples, random seed 42, seed-level resampling,
    and statement that trials are not treated as independent bootstrap units
    for seed-level CIs.
- [x] Include ledger schema table.
  - Required fields: `trial_id`, `campaign_id`, `budget_index`, `decision`,
    `failure_category`, `proposal`, `patch_ref`, `raw_log_ref`,
    `parsed_metrics`, `provenance`, `node_state_hash`.
  - Requirement: distinguish terminal lifecycle states from failure labels.
- [x] Include lifecycle pseudocode.
  - Required steps: open pending trial, request proposal, validate edit scope,
    reject no-op or invalid preconditions, apply patch, run worker, parse metric,
    decide keep/discard/failed-invalid, append ledger, update memory context.
- [x] Include provenance/audit examples.
  - Include: one successful kept trial, one discarded trial, one failed-invalid
    trial, and one stress/no-op or scope-violation trial.
  - Requirement: each example should identify ledger path and artifact refs, not
    paste long raw logs.

## 4. Notebook And Plot Policy

- [x] Include the notebook as an artifact, not as the primary evidence source.
  - File: `paper_figures.ipynb`.
  - Requirement: notebook regenerates plots/tables from ledgers and tables.
  - Requirement: notebook overview explains the evidence set, output paths, and
    which figures are main-paper, supplement-only, or retired.
  - Install path: `uv pip install -e ".[dev,notebook]"`.
- [x] In `supplement.tex`, describe the notebook briefly.
  - Include: notebook filename, input directories, output directories, and run
    instruction (`Kernel -> Restart & Run All`).
  - Requirement: notebook is an analysis convenience layer; append-only ledgers
    remain the source of truth.
- [x] Include only selected plots in the supplement.
  - Include: plots that support auditability, provenance, full OpenML detail,
    full L40S seed detail, or reproducibility checks.
  - Do not include: every exploratory plot, stale MLP-only plots, or notebook
    figures that are not tied to a submitted claim.
- [x] Mark retired or exploratory plots in the notebook.
  - Requirement: MLP synthetic plots are ledger-only/exploratory unless the
    paper explicitly restores MLP to the six-node evidence set.
  - Requirement: old ResNet 135-trial plots must be labelled as 3-converging-seed
    diagnostics if retained.

## 5. DeepSeek/API Disclosure

- [x] Add hosted-model/API disclosure.
  - Include: manager/proposal model identifier
    `deepseek/deepseek-v4-flash`.
  - Include: access mode/API provider and approximate date range if known.
  - Include: temperature and other generation settings used by each campaign
    class.
  - Requirement: do not include API keys, account IDs, or private endpoint
    secrets.
- [x] Add reproducibility caveat for hosted APIs.
  - Requirement: state that exact proposal text may change if the hosted model
    changes, but governance metrics are recomputable from submitted ledgers.
  - Include: local/stub or deterministic-worker paths where available.

## 6. Repository Organization Plan

### 6.1 Submission Package Boundary

- [ ] Choose one canonical paper source directory.
  - Current issue: active paper source is the submodule
    `A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/`, while
    older paper material also exists under `paper/kdd_aae_2026/`.
  - Requirement: mark one as canonical in root `README.md`; archive or clearly
    label the other to avoid reviewers building the wrong paper.
- [ ] Create a top-level `repro/` or `artifact/` index.
  - Include: `README.md`, `MANIFEST.md`, `ENVIRONMENT.md`,
    `RUNBOOK.md`, `ANONYMIZATION.md`.
  - Requirement: the index should tell reviewers what to run locally, what was
    run on `deepthought2`, and what requires hosted-model access.
- [ ] Add a submitted-state manifest.
  - Include: code commit, paper commit/submodule commit, table checksums,
    figure checksums, ledger counts, and artifact-bundle checksum.

### 6.2 Source Tree Cleanup

- [ ] Keep source code under `src/autoresearch/`.
  - Requirement: do not mix experimental outputs into source directories.
- [ ] Keep node contracts under `configs/nodes/`.
  - Requirement: each submitted node has a matching README or supplement entry
    explaining editable paths, frozen paths, run command, metric parser, and
    budget.
- [ ] Keep experiment nodes under `nodes/`.
  - Requirement: generated state files, large datasets, checkpoints, logs, and
    runtime artifacts remain ignored or moved to the artifact bundle.
- [ ] Clean generated local files before packaging.
  - Include: `.DS_Store`, `__pycache__/`, `.pytest_cache/`, local `.venv/`,
    LaTeX byproducts, notebook checkpoints.
  - Requirement: do not delete user data; remove only generated clutter after
    confirming it is ignored or archived.

### 6.3 Evidence And Data Layout

- [ ] Separate ledgers from bulky artifacts.
  - Keep: submitted ledgers and event streams needed to recompute tables.
  - Archive: raw logs, patch diffs, checkpoints, large datasets, and `/ceph`
    server artifacts in a separate bundle.
- [ ] Create `experiments/README.md`.
  - Include: which ledger files belong to the 1,445-trial evidence set.
  - Include: which ledgers are smoke/dev/retired.
  - Include: how to validate trial counts and provenance completeness.
- [ ] Create `paper/tables/README.md` and `paper/figures/README.md`.
  - Include: generating script/notebook for each table/figure.
  - Requirement: each generated artifact has one authoritative source command.

### 6.4 Scripts And Runbooks

- [ ] Split scripts into categories or document them clearly.
  - Suggested categories: `run_*` campaign launchers, `analyze_*` analysis,
    `export_*` paper tables/figures, `check_*` validation, `reset_*` node state.
- [ ] Add a minimal reproduction path.
  - Include: install dependencies, run tests, inspect node spec, run one
    smoke campaign, regenerate paper tables from submitted ledgers.
  - Requirement: this path should run on a laptop without `deepthought2`.
  - Include notebook dependencies through the `notebook` optional extra.
- [ ] Add a full reproduction path.
  - Include: server requirements, GPU requirements, DeepSeek/API requirements,
    expected runtime, and artifact storage path.
  - Requirement: identify which experiments cannot be cheaply rerun during
    review.

### 6.5 Tests And Validation

- [ ] Add a single validation command for the submitted artifact.
  - Suggested command: `pytest` plus a script that checks ledger counts,
    provenance completeness, table regeneration, and known campaign IDs.
- [ ] Add or update tests for paper-facing invariants.
  - Include: 1,445 total governed trials, 100% provenance completeness across
    submitted nodes, failure taxonomy names, no stale paper refs, and no abstract
    memory-performance claim.
- [ ] Add reproducibility status badges or checklist output only after tests are
      stable.

### 6.6 README Updates

- [ ] Update stale headline counts.
  - Current root README says 1,355 reported governed trials; paper now uses
    1,445 governed trials.
- [ ] Add "start here" instructions for reviewers.
  - Include: paper source, supplement source, artifact manifest, notebook, and
    validation commands.
- [ ] Add a privacy/anonymization note.
  - Include: what is withheld during review and what will be released after
    acceptance.

## 7. Notebook Update Checklist

- [x] Add current project overview to `paper_figures.ipynb`.
- [x] Filter notebook analysis to the submitted campaign set by default.
  - Requirement: exclude smoke, dev, retired MLP-only, and stale environment
    failure runs from main-paper figures.
- [x] Add a visible "source of truth" note.
  - Requirement: ledgers and generated CSVs are source data; notebook plots are
    derived views.
- [x] Label each figure as one of: main paper, supplement candidate, retired,
      or exploratory.
- [x] Replace stale 135-trial ResNet text with current 225-trial/caveated text
      or clean-rerun text once final data are synced.
- [x] Add an output manifest cell.
  - Include: filename, source ledger/table, paper section, and status.
- [x] Clear or minimize notebook outputs before artifact packaging unless
      rendered outputs are intentionally included.
