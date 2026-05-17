# Cleanup Candidates

Date: 2026-05-12

This is a candidate list only. Nothing here has been deleted. Use it as a
review checklist before a cleanup pass.

Risk levels:

- Low: generated, ignored, empty, or clearly superseded.
- Medium: useful historical evidence, but not needed for current KDD paper
  progress.
- High: tracked source/history or files referenced by docs/tests; clean only
  after deciding to rewrite references.

## Keep as Current Working Truth

Do not clean these while preparing final evidence:

- `plan/Up-to-date.md`
- `plan/TODO.md`
- current source code under `src/`, `scripts/`, `configs/`, `tests/`
- `experiments/ledgers/kdd_main_5trial_trials.jsonl`
- `experiments/events/kdd_main_5trial_events.jsonl`
- `experiments/ledgers/ablation_none_trials.jsonl`
- `experiments/ledgers/ablation_append_only_summary_trials.jsonl`
- `experiments/ledgers/ablation_append_only_summary_with_rationale_trials.jsonl`
- `experiments/events/ablation_none_events.jsonl`
- `experiments/events/ablation_append_only_summary_events.jsonl`
- `experiments/events/ablation_append_only_summary_with_rationale_events.jsonl`
- `experiments/artifacts/ablation_none/`
- `experiments/artifacts/ablation_append_only_summary/`
- `experiments/artifacts/ablation_append_only_summary_with_rationale/`
- `experiments/ledgers/kdd_stress_scope_trials.jsonl`
- `experiments/ledgers/kdd_stress_noop_trials.jsonl`
- `experiments/events/kdd_stress_noop_events.jsonl`

## Low-Risk Markdown Cleanup Candidates

These are stale status/planning docs now superseded by `plan/Up-to-date.md`
and `plan/TODO.md`. Prefer archiving instead of deleting if you want provenance.

| Path | Why stale |
|---|---|
| `paper_preparation.md` | Still says main and memory evidence are dry-run. Current ablation ledgers are real; status is superseded. |
| `paper/notes/current_readiness_next_steps.md` | Still says memory ablation is incomplete and has a stale pending guard. Current ledgers have 5 real records per arm and no pending guards. |
| `plan/KDD_AAE_refinement_plan_v2.md` | Still centers `resnet_real_incremental` Level 1 evidence and old TODO status. Good framing, stale status. |

Suggested cleanup action:

- Move these into `plan/archive/` or add a short deprecation note at the top.

## Medium-Risk Markdown Cleanup Candidates

These are historical planning drafts. They may be useful for provenance but are
not necessary as active instructions.

| Path | Notes |
|---|---|
| `plan/archive/raw_plan_stage2.md` | Early Stage 2 implementation plan; many statuses are now obsolete. |
| `plan/archive/kdd_aae_refinement_plan_md.md` | Superseded by v2 and now by `Up-to-date.md`; long and status-stale. |
| `plan/archive/stage2-summary.md` | Historical deliverable summary. |
| `plan/archive/stage2_real_resnet_report.md` | Useful evidence for old `resnet_real_incremental`, but not current main result. |
| `plan/archive/checklist_stage3.md` | Historical checklist, partly overtaken by current TODO. |
| `plan/archive/limitations.md` | Says real memory ablation still depends on stable runner; stale. |
| `plan/archive/contribution_claims.md` | Contains memory-improvement wording that current evidence does not support. |
| `plan/archive/related_work.md` | Minimal notes, likely superseded by paper sections and literature synthesis. |

## Duplicate Markdown Candidate

These two files are byte-identical:

- `docs/stage2/stage_2_kdd_aae_instruction_aligned.md`
- `plan/archive/stage_2_kdd_aae_instruction_aligned.md`

Suggested cleanup action:

- Keep the `docs/stage2/` copy if it is active documentation.
- Remove or archive-note the `plan/archive/` duplicate.

## Paper Drafts and Tables That Are Stale

These should not necessarily be deleted, but they should be considered stale
until regenerated or rewritten from the final evidence set.

| Path | Why stale |
|---|---|
| `paper/kdd_aae_2026/sections/05_results.md` | Describes main campaign and ablation as dry-run governance-contract ledgers. That no longer matches current ablation evidence. |
| `paper/kdd_aae_2026/sections/04_experiments.md` | Mentions `baseline_manager` in current dry-run artifacts; should be rewritten after final run choice. |
| `paper/kdd_aae_2026/sections/06_discussion_limitations.md` | Current “what is proven” section still depends on dry-run framing. |
| `paper/tables/*.csv` | Tables are not synchronized with current ledgers. Regenerate after final evidence. |
| `paper/tables/*.txt` | Reports are stale, especially memory ablation and artifact completeness. |
| `paper/figures/*.svg` | Figures reflect older dry-run or old ResNet evidence. Regenerate after final evidence. |
| `paper/figures/*.csv` | Older figure source CSVs from 2026-04-30. |
| `artifact_manifest.json` | Generated 2026-05-10 and does not reflect current final-evidence decision. Regenerate later. |

## Low-Risk Untracked Experiment Ledgers

These are ignored/generated and not tracked by git. Most are smoke or superseded
runs. Keep only if you want a forensic trail.

| Path | Reason |
|---|---|
| `experiments/ledgers/kdd_main_5trial_v2_trials.jsonl` | Unreferenced outside itself; dry-run worker. |
| `experiments/ledgers/ollama_native_smoke_trials.jsonl` | One-off smoke, no external refs found. |
| `experiments/ledgers/ollama_langchain_smoke_trials.jsonl` | One-off smoke, only referenced by stale `paper_preparation.md`. |
| `experiments/ledgers/ollama_real_smoke_trials.jsonl` | One-off smoke, only referenced by stale `paper_preparation.md`. |
| `experiments/ledgers/real_smoke_v3_trials.jsonl` | One-off verification run; only script docs use the id as an example. |
| `experiments/ledgers/smoke_append_only_summary_trials.jsonl` | One-record dry-run smoke. |
| `experiments/ledgers/smoke_append_only_summary_with_rationale_trials.jsonl` | One-record dry-run smoke. |
| `experiments/ledgers/smoke_none_trials.jsonl` | One-record dry-run smoke; runner uses the name as an example. |
| `experiments/ledgers/manager_comparison_baseline_manager_trials.jsonl` | Optional manager comparison output; not part of current primary evidence. |
| `experiments/ledgers/manager_comparison_prompt_manager_trials.jsonl` | Optional manager comparison output; not part of current primary evidence. |

## Medium-Risk Tracked Legacy Experiment Ledgers

These are tracked by git, so deleting them is a repository-history decision.
They look stale relative to current KDD evidence, but some docs/tests still
reference them.

| Path | Notes |
|---|---|
| `experiments/ledgers/resnet_real_3x5_memory_packet_trials.jsonl` | Older 15-record run; referenced by old figure CSVs. |
| `experiments/ledgers/resnet_real_3x5_stage2_trials.jsonl` | Older 15-record run; no active reference found outside ledger. |
| `experiments/ledgers/resnet_real_baseline_fallback_trials.jsonl` | Old one-record run; manifest references it. |
| `experiments/ledgers/resnet_real_clean_round2_data_trials.jsonl` | Old one-record run; manifest references it. |
| `experiments/ledgers/resnet_real_incremental_trials.jsonl` | Historical accepted-edit demo; referenced by old report and refinement plans. |
| `experiments/ledgers/resnet_real_second_round_trials.jsonl` | Old two-record run; no active reference found outside ledger. |
| `experiments/ledgers/resnet_trigger_*_ablation_trials.jsonl` | Older 9-record ablation ledgers; superseded by current `ablation_*` ids. |
| `experiments/ledgers/stage2_full_round_trials.jsonl` | Older 15-record Stage 2 round; manifest references it. |
| `experiments/ledgers/smoke2_trials.jsonl` | Old smoke ledger. |
| `experiments/ledgers/smoke_test_trials.jsonl` | Old smoke ledger referenced by docs/notebooks. |
| `experiments/ledgers/langgraph_smoke_trials.jsonl` | Old smoke ledger referenced by README/notebooks. |

Suggested cleanup action:

- Move old tracked ledgers into `experiments/ledgers/archive/` only if you also
  update references and tests.
- Otherwise leave them until the paper evidence set is finalized.

## Low-Risk Experiment Artifact Directories

These artifact directories are empty and ignored. They can be removed without
losing data:

- `experiments/artifacts/kdd_main_5trial/`
- `experiments/artifacts/kdd_stress_noop/`
- `experiments/artifacts/kdd_stress_scope/`
- `experiments/artifacts/ollama_langchain_smoke/`
- `experiments/artifacts/ollama_native_smoke/`
- `experiments/artifacts/ollama_real_smoke/`
- `experiments/artifacts/resnet_real_3x5_memory_packet/`
- `experiments/artifacts/resnet_real_3x5_stage2/`
- `experiments/artifacts/resnet_real_baseline_fallback/`
- `experiments/artifacts/resnet_real_clean_round2_data/`
- `experiments/artifacts/resnet_real_incremental/`
- `experiments/artifacts/resnet_real_second_round/`

Also likely removable:

- `experiments/artifacts/trial-001/generated_packet.json`

Reason: it appears to be an orphan smoke artifact. Docs and notebooks mention
`experiments/artifacts/trial-001/...` generically, but the current directory
contains only `generated_packet.json`, not the expected patch/log pair.

## Event Streams

Keep current event streams for now:

- `experiments/events/kdd_main_5trial_events.jsonl`
- `experiments/events/ablation_none_events.jsonl`
- `experiments/events/ablation_append_only_summary_events.jsonl`
- `experiments/events/ablation_append_only_summary_with_rationale_events.jsonl`
- `experiments/events/kdd_stress_noop_events.jsonl`

Low-risk candidates if the matching smoke ledgers are removed:

- `experiments/events/ollama_langchain_smoke_events.jsonl`
- `experiments/events/ollama_native_smoke_events.jsonl`
- `experiments/events/ollama_real_smoke_events.jsonl`
- `experiments/events/real_smoke_v3_events.jsonl`

## Node-Local Runtime State

These are generated runtime files under `nodes/ResNet_trigger/`. They are
ignored by the node `.gitignore` and should be cleared before final campaigns:

- `nodes/ResNet_trigger/.autoresearch_state.json`
- `nodes/ResNet_trigger/experiment_memory.jsonl`
- `nodes/ResNet_trigger/results.tsv`
- `nodes/ResNet_trigger/run.log`

Large ignored node-local artifact candidates:

| Path | Size | Notes |
|---|---:|---|
| `nodes/ResNet_trigger/artifacts/pre_sync_backup_20260409_233522/` | 11M | Old backup. |
| `nodes/ResNet_trigger/artifacts/pre_sync_backup_20260409_clean5_000136/` | 8.4M | Old backup. |
| `nodes/ResNet_trigger/artifacts/pre_sync_backup_20260410_3x5_010840/` | 8.6M | Old backup. |
| `nodes/ResNet_trigger/artifacts/run_archive/` | 1.9M | Old local run archive; useful only for forensic inspection. |
| `nodes/ResNet_trigger/artifacts/notebook_best_model.pt` | 3.7M | Old notebook checkpoint. |
| `nodes/ResNet_trigger/artifacts/best_model.pt` | 8.1M | Latest node-local checkpoint; generated, not paper source of truth. |

Keep the source HDF5 data unless you know it can be recreated:

- `nodes/ResNet_trigger/noise_traces_4000x8000.h5`
- `nodes/ResNet_trigger/signal_vacuum_sum_crop_4000x8000.h5`

## Legacy Node Cleanup Candidates

`nodes/autoresearch-macos/` appears to be a Stage 1 / legacy node. It is mostly
tracked, so cleanup here is a project-structure decision, not simple cache
cleanup.

Low-risk generated files:

- `nodes/autoresearch-macos/run.log`
- `nodes/autoresearch-macos/__pycache__/`
- `nodes/autoresearch-macos/.venv/` (about 616M, ignored)

Medium/high-risk tracked legacy state:

- `nodes/autoresearch-macos/.autoresearch_state.json`
- `nodes/autoresearch-macos/experiment_memory.jsonl`
- `nodes/autoresearch-macos/results.tsv`
- `nodes/autoresearch-macos/memory/*.md`

These are likely not needed for the current KDD Stage 2 paper, but deleting
tracked legacy node files should be a deliberate deprecation commit.

## General Generated Junk

Low-risk cleanup patterns:

- `.DS_Store` files throughout the repo.
- `.pytest_cache/`
- `pytest-cache-files-9oneolqk/`
- `__pycache__/` directories under `scripts/`, `src/`, `tests/`, and nodes.
- root `.venv/` if it is not the active development environment.
- `nodes/ResNet_trigger/.venv/` if it is not actively used.

## Submodules

Do not clean inside these as part of the main repo cleanup:

- `harness/claw-code/`
- `A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/`

`harness/claw-code` currently appears as a modified submodule pointer in the
parent repo. Treat it as its own repository.
