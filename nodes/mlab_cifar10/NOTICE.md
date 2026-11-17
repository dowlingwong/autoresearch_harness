# Node mlab_cifar10 — origin and changes

Ported for Track G (Paper B, R1) from **MLAgentBench** (https://github.com/snap-stanford/MLAgentBench,
MIT licence, Copyright (c) the MLAgentBench authors), task `benchmarks/cifar10` at the commit the
port read (see the port report). The suite's `train.py` was restructured into the Track G fixed-step API:
the editable `train.py` keeps the suite's model, optimizer, loss and preprocessing (see its docstring for
every difference); the training loop, step budget, data handling, validation split and metric moved into
the frozen files `run_fixed_budget.py`, `ar_fixed_loop.py`, `ar_data.py` and `prepare_data.py`, written
by the port. The task data are not part of the node; `prepare_data.py` fetches them from their public
sources (their own terms apply) and records their hashes.
