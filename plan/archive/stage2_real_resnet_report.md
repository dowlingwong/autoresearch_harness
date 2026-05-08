# Stage 2 Real ResNet-Trigger Report

## Summary

Stage 2 now has a real ResNet-trigger full-loop demonstration through the governed control plane. The run used the Stage 2 `run_real_campaign` path, a `baseline_manager` proposal, the `ClawWorker` backend, Ollama/Qwen for the worker edit step, and the ResNet-trigger node training command.

The clean paper-facing campaign is:

- Campaign id: `resnet_real_incremental`
- Node: `resnet_trigger`
- Manager: `baseline_manager`
- Memory mode: `append_only_summary_with_rationale`
- Worker: `claw_style_worker`
- Measurements: 1 real original-config baseline + 1 real agent trial
- Edit: `LEARNING_RATE = 1e-3` to `LEARNING_RATE = 5e-4`
- Decision: `kept`
- Final metric: `val_auc = 0.782756`
- Baseline metric: `val_auc = 0.779911`
- Incremental gain: `+0.002845`
- Validity: `valid`

## Why This Matters

This run demonstrates the Stage 2 claim: the project is not only a coding-agent loop. It is a governed autonomous experimentation harness that records proposals, worker artifacts, edit scope, parsed metrics, validity, decisions, and provenance in a paper-facing ledger.

The real run exercised:

- manager proposal generation
- generated worker packet
- bounded `train.py` edit
- real ResNet-trigger training command
- metric parsing from `run.log`
- editable-scope validation
- keep/discard decision in the Stage 2 control plane
- append-only JSONL ledger
- artifact capture for audit

## Campaign Results

From `paper/tables/main_campaign_summary.csv`:

| Metric | Value |
|---|---:|
| Initial val_auc | 0.779911 |
| Best val_auc | 0.782756 |
| Final accepted val_auc | 0.782756 |
| Net gain | 0.002845 |
| Total records | 2 |

From `paper/tables/governance_metrics.csv`:

| Governance metric | Value |
|---|---:|
| Kept trials | 2 |
| Discarded trials | 0 |
| Failed invalid trials | 0 |
| Acceptance rate | 1.0 |
| Invalid rate | 0.0 |
| Complete provenance rate | 1.0 |
| Editable-scope violations | 0 |
| Command failure rate | 0.0 |
| Metric parsing failure rate | 0.0 |
| Artifact capture completeness | 1.0 |

## Trial Details

The baseline record is `resnet_real_incremental-trial-000`.

- Original configuration: `LEARNING_RATE = 1e-3`
- Baseline metric: `val_auc = 0.779911`

The accepted agent trial is `resnet_real_incremental-trial-001`.

The manager proposal was:

> Change `LEARNING_RATE` from `1e-3` to `5e-4` and keep all other hyperparameters unchanged.

The captured patch is:

```diff
-LEARNING_RATE = 1e-3
+LEARNING_RATE = 5e-4
```

The final run log reported:

| Run metric | Value |
|---|---:|
| val_bpb | 0.217244 |
| val_auc | 0.782756 |
| val_pr_auc | 0.795560 |
| val_roc_auc | 0.782756 |
| epochs_completed | 3 |
| training_seconds | 118.9 |
| total_seconds | 140.5 |
| num_steps | 66 |
| num_params_M | 0.962 |

Epoch-level validation AUC:

| Epoch | val_auc |
|---:|---:|
| 1 | 0.782756 |
| 2 | 0.780756 |
| 3 | 0.782000 |

Early stopping triggered after epoch 3 because the current validation AUC did not improve beyond the configured minimum delta.

## Incremental Result

| Step | Configuration | val_auc | Decision |
|---:|---|---:|---|
| 0 | original config, `LEARNING_RATE = 1e-3` | 0.779911 | kept baseline |
| 1 | agent edit, `LEARNING_RATE = 5e-4` | 0.782756 | kept |

The real agent trial improved validation AUC by `+0.002845` over the real baseline run under the same fast CPU smoke configuration.

## Generated Artifacts

Primary artifacts:

- `experiments/ledgers/resnet_real_incremental_trials.jsonl`
- `experiments/artifacts/resnet_real_incremental/trial-001/generated_packet.json`
- `experiments/artifacts/resnet_real_incremental/trial-001/patch.diff`
- `experiments/artifacts/resnet_real_incremental/trial-001/run.log`
- `experiments/artifacts/resnet_real_incremental/trial-001/parsed_metrics.json`
- `experiments/artifacts/resnet_real_incremental/trial-001/legacy_loop_result.json`

Paper tables and figure data:

- `paper/tables/main_campaign_summary.csv`
- `paper/tables/governance_metrics.csv`
- `paper/figures/campaign_trajectory.csv`
- `paper/figures/accepted_discarded_invalid_counts.csv`
- `paper/figures/gain_per_budget_unit.csv`
- `paper/figures/repeated_bad_idea_rates.csv`

Rendered SVG plots:

- `paper/figures/resnet_real_incremental_trajectory.svg`
- `paper/figures/resnet_real_incremental_decisions.svg`
- `paper/figures/resnet_real_incremental_epoch_val_auc.svg`

The notebook that regenerates these report plots is:

- `notebooks/stage2_real_resnet_report.ipynb`

## Interpretation

The result should be framed as a real full-loop systems demonstration, not as an optimization win. A one-trial campaign cannot establish search effectiveness. It does establish that Stage 2 can run a real ML experiment, capture artifacts, validate edit scope, parse metrics, and record an authoritative keep decision.

The incremental baseline-plus-agent run is more convincing than a first-valid-trial demonstration because it records a real pre-edit metric and then shows the Stage 2 control plane accepting a bounded agent edit with a positive delta.

For the Stage 2 report, the strongest claim is:

> Stage 2 converts an autoresearch-style worker loop into an auditable, governed experiment protocol with explicit node contracts, artifact capture, validity checks, and paper-facing metrics.

## Limitations

- The clean campaign has one baseline measurement and one agent trial; it is still small.
- Net gain is positive, but one candidate trial is not enough to claim robust optimization.
- The worker backend required a Stage 2 fallback path because the legacy worker sometimes edited `train.py` but reported no changed files.
- The real run used a fast CPU smoke configuration, not the full overnight campaign budget.
- The current paper evidence should be extended with a multi-trial real campaign and memory ablation.

## Recommended Next Experiments

- Run a 5-trial real campaign from a clean node state.
- Add a manager that avoids no-op repeats when the first baseline change is already present.
- Run real ablations for:
  - `none`
  - `append_only_summary`
  - `append_only_summary_with_rationale`
- Include one forced invalid-scope or failed-command trial to demonstrate governance behavior under failure.
- Add an additional node or task family before making broader benchmark claims.
