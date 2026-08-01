# Reliability configuration directory

Machine-readable configurations for the reliability scaffold. Each example
config carries `status: synthetic_fixture` or an explicit example marker and
must not be used for confirmatory runs.

| Subdirectory | Purpose |
| --- | --- |
| `schemas/` | JSON Schema exports generated from `src/autoresearch/reliability/schema/` |
| `faults/` | Fault manifests (sealed before each run; policy-visible subset only) |
| `tasks/` | Task manifests (real tasks added after pilot authorization) |
| `splits/` | Group-isolated split manifests (frozen after leakage audit) |
| `experiments/` | Experiment configurations that bind task/fault/split/model |

The `R1_P_synthetic.yaml` experiment is the fixture used by
`scripts/run_r1_synthetic_readiness.py`. It exercises the scaffold with
synthetic inputs and does not claim any pilot gate has passed.
