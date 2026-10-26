# Experimental MLGym adapter: Battle of Sexes

This scaffold maps the upstream `battleOfSexes` task's JSON `Score` (higher
row-player payoff is better) to the harness metric and editable scope. It is
not provisioned for real execution: `run_mlgym.py` always fails, and
`strategy.py` is only a placeholder. No task code or evaluation data is
vendored. The callable adapter translates the `(metrics, submission)` result
from `PythonSubmissionTasks.evaluate`; a fake result is used only in tests.

Source inspected: https://github.com/facebookresearch/MLGym, commit
`9d40c1b5035202018cd7091fb4e83a9c68b377c0`, `configs/tasks/battleOfSexes.yaml`,
`data/battleOfSexes/evaluate.py`, and `mlgym/environment/tasks.py`. The task
requires editing `row_strategy(history)` in `strategy.py`; the evaluator emits
a JSON object with `Score` from 10 rounds and 10,000 simulations. Its
`evaluate.py` is read-only under the upstream task config. MLGym's source is
CC-BY-NC-4.0; this scaffold does not redistribute its code.

Before any actual task run, the author must resolve container access (G.2),
select/approve tasks and their role and budget, and establish a bridge that
provisions the pinned upstream workspace, preserves the read-only evaluator,
runs its environment, and captures the authentic evaluation result. Candidate
and evaluator isolation, seed/reproducibility and compute must be checked in
that environment. This synthetic smoke establishes none of those properties.

The focused test uses `FakeMLGymWorker` with an injected fake evaluator and a
fake manager. It runs the governed campaign loop for three trials; each proposed
path is checked before the fake evaluator is called, and each returned value
passes through `adapt_evaluation` and `parse_score`. Its synthetic patch is an
illustrative record only and does not edit or execute a task workspace. The test
uses the harness function named `run_real_campaign` solely to exercise the
control plane with this explicitly fake worker. This does not implement or test
MLGym container integration.
