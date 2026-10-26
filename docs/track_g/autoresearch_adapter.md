# Track G autoresearch adapter

`track_g_autoresearch` is a LocalWorker adapter for the bundled
`nodes/autoresearch-macos` source. It emits `train.py` stdout to the worker
instead of redirecting it; the worker stores the captured log under the trial
artifacts. The metric is `val_bpb`, minimized. `prepare.py` fixes the training
budget at 300 seconds, excluding startup/compilation; evaluation takes
additional time. The existing macOS and Linux configurations remain intact.

When authorized on a suitable GPU host, supply the node root explicitly and
allow enough time for startup, training and evaluation. The intended
`scripts/run_campaign.py` route requires `--node track_g_autoresearch`,
`--node-root nodes/autoresearch-macos`, and `--worker local`; without the
last option it selects ClawWorker. This runner does not accept
`--timeout-seconds`: LocalWorker uses its 3600-second default. It also requires
`--campaign-id` and `--budget`. The alternative
`scripts/run_local_node_campaign.py` always uses LocalWorker and does accept
`--timeout-seconds`, but defaults to 240 seconds, less than the training
budget alone. Choose a timeout appropriate to the whole run.

Environment setup, data preparation, host readiness, and a suitable manager
must be checked separately before a real run. This config is for LocalWorker;
ClawWorker expects its own `run.log` convention. Neither runner invocation is
authorization to run real training. The CLI behavior above was checked by
reading the argument parser and worker construction; it has not been executed
against a real node.

The local smoke uses a stub `train.py` and a command override, exercises three
proposals and keep/discard decisions, and does not run model training. The
adapter-specific editor supports commented constant assignments in the bundled
script and rejects stale old values and invalid Python; the script itself and
existing worker behavior for other nodes are unchanged.

The current script hard-codes `torch.manual_seed(42)` and does not consume the
worker seed variables. Training-seed and split-seed control and the design of
G.3 noise-floor measurements remain unresolved; this smoke supplies no noise,
GPU, runtime, reproducibility, benchmark-identity, or scientific evidence.
The author still decides Track G node roles and audit budgets (G.4).
