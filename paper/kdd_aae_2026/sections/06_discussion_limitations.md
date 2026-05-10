# 6. Discussion and Limitations

## What Is Proven

The current artifact suite proves that the control-plane protocol can run a bounded fixed-budget campaign, classify keep/discard/failed-invalid outcomes, preserve append-only provenance, recover per-trial artifact references, and expose governance metrics as paper tables and figures. It also proves that invalid actions can be represented as first-class audit objects: the stress trial records `failed_invalid / invalid_edit_scope` without committing forbidden state.

The real ResNet full-loop demonstration proves that the harness can execute a meaningful scientific ML experiment, parse validation AUC, and keep an improving code edit. The KDD dry-run campaign suite proves that the governance/reporting contract scales across the planned fixed-budget tables, figures, manifest, and artifact-completeness checks.

## What Is Not Proven

This work does not claim to build a general autonomous scientist, prove scientific discovery, introduce a universal optimization algorithm, or depend on a specific coding-agent backend. The ResNet-trigger task is used as a real scientific ML case study for evaluating governed autonomous experimentation; it is not claimed to represent all ML optimization tasks.

The current artifacts do not prove broad generalisation across many task nodes, memory effects at larger real-worker scale, or manager superiority. The manager comparison shows that governance metrics can be produced under multiple manager modes, but it is not a claim that one manager is universally better.

## Why Governance Metrics Matter

Final AUC alone cannot answer whether an autonomous experiment was scientifically credible. An agent could improve a metric while repeatedly attempting invalid edits, hiding failed runs, corrupting state, or keeping decisions that cannot be audited. Governance metrics make those behaviors visible. Acceptance rate, invalid rate, repeated-bad rate, provenance completeness, artifact capture completeness, and failure categories describe the reliability of the experimentation process around the model.

This distinction is the main contribution. The harness does not replace task metrics; it makes task metrics interpretable by attaching lifecycle, validity, and provenance context to every trial.

## Limitations

We evaluate on one real scientific ML node. This demonstrates real governed execution, but does not claim broad benchmark coverage.

All experiments use the ResNet-trigger node. We cannot rule out that reported improvements overfit to this evaluation domain. Applying the harness hill-climbing methodology of Trivedy (2026) across multiple evaluation nodes with holdout splits would strengthen the governance claims.

Dry-run tests validate control-plane contracts; reported empirical results use real worker campaigns unless explicitly marked as smoke tests.

Orchestration backends, cloud execution, and UI layers are future extensions. This work focuses on the control-plane protocol and its audit metrics.

## Generalisation Path

The NodeSpec YAML pattern generalises the harness to new ML experiments without code changes; each spec is a harness template for a class of experiments. A stronger follow-up evaluation would add multiple node specs, split them into optimisation and holdout nodes, and report whether governance metrics remain stable when the task domain changes.
