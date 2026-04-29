# Architecture

The Stage 2 system is a governed autonomous experimentation framework.

Core layers:

1. Manager proposes bounded experiment changes.
2. Control plane owns lifecycle, budget, pending-trial guards, decisions, and records.
3. Worker runtime applies bounded code changes and returns structured results.
4. Experiment node declares editable paths, commands, metrics, and validity rules.
5. Memory/audit layer stores append-only trial records and compressed manager context.
6. Evaluation/reporting layer exports deterministic paper metrics and CSVs.

The canonical real campaign path is:

```text
ManagerProposal
  -> Stage 2 run_real_campaign
  -> pending-trial guard
  -> ClawWorker
  -> harness/claw-code autoresearch loop
  -> WorkerResult
  -> Stage 2 scope/metric/decision logic
  -> append-only TrialRecord
```

`harness/claw-code` is the current worker backend behind the Stage 2 control
plane. It is not the authoritative campaign controller for Stage 2. The legacy
loop may recommend keep/discard internally, but Stage 2 ignores that as an
authority and makes the final decision from changed files, parsed metrics, node
contract, budget state, and campaign history.

For real campaigns, Stage 2 captures:

- generated worker packet
- patch diff when available
- raw run log when available
- parsed metric payload
- worker commit before/after
- raw legacy loop result
- manager proposal metadata such as LangGraph prompt/output hashes

The core contribution is the control plane, audit model, and evaluation surface,
not the specific worker backend.
