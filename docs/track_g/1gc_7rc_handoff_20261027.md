# 1GC-7RC light tasks: source and implementation status

AI-assisted. Status: BLOCKED; no runnable benchmark adapter or smoke campaign is claimed.

The paper's [v2 §3.2–3.4 and Table 2](https://arxiv.org/html/2605.17046v2) describes the following 40-minute tasks. Its evaluation split is validation for T1 and T7 and held-out test for T5 and T6. These are benchmark task budgets, not an approved Paper B audit budget.

| Task | Dataset | Paper metric | Direction | Scoring split |
|---|---|---|---|---|
| T1 | TinyShakespeare | validation bits per byte | minimize | validation |
| T5 | Forest Cover | accuracy | maximize | test |
| T6 | ETTh1 | mean squared error | minimize | test |
| T7 | AG News | accuracy | maximize | validation |

The paper reports a locked `prepare.py`, a baseline `train.py`, an isolated evaluator, scored checkpoints, a `build_model(config)` interface, and JSON evaluation output. The paper does not give the exact task-specific evaluator output keys, checkpoint format, CLI arguments, file layout, dependencies, or source digests. It states that agents can create `run_{x}.py` files; Paper B RUNBOOK G.1 instead requests `train.py` editable and `prepare.py` locked. That narrower scope is a Paper B adaptation and should not be called an unchanged upstream agent protocol.

On 2026-10-27 the paper's stated repository https://github.com/Strolchii/1GC-7RC-Benchmark returned 404 through the web fetch; GitHub repository search for `1GC-7RC` returned no repositories. No upstream source revision, actual locked evaluator, or baseline was available for inspection. The article alone cannot establish an executable adapter or verify a parser against actual output. No node specs or synthetic `prepare.py` have been registered, because this would imply support without the benchmark evaluator. No proposals or 3-trial stub campaigns were run; L4.1's done criterion remains unmet.

To unblock: obtain an authoritative, revision-identified copy of all four upstream task directories and license terms, verify evaluator and data/checkpoint contracts and hashes, and implement isolated scoring and scope enforcement before exercising stub campaigns. Author decisions remain required for Track G role and audit budget (RUNBOOK G.4), and any Paper B deviations from upstream should be documented as such. In particular, clarify whether Paper B intends its narrower edit scope and how upstream's held-out test scoring for T5/T6 is to be used in repeated proposal selection. No conclusion about scientific readiness or permission for remote work follows from this note.
