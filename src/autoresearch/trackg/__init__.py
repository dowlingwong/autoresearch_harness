"""Track G shared pieces for porting real GPU tasks (M10 steps 0-1, R1-BX-D4/D5).

This package is additive: nothing in the T0 path (``manager_proposal/v1``,
``run_real_campaign``, ``LocalWorker``) imports it.

Modules
-------
- :mod:`node_template` -- the Track G ``NodeSpec`` YAML (``trackg_node/v1``)
  with a fixed step or token budget; a wall-clock budget is refused.
- :mod:`outcome` -- the base metric parser; exactly one of ``ok(value)``,
  ``divergent`` or ``infrastructure_failure`` per run.
- :mod:`determinism` -- deterministic settings and the record of what could
  not be forced.
- :mod:`runner` -- the per-run working-copy runner with resource limits.
- :mod:`selection` -- validation-only selection for test-scored tasks.
- :mod:`diff_support` -- unified-diff application through the existing scope
  validator, inert and duplicate detection, provenance (G.2).
- :mod:`proposal_v2_diff` -- the ``manager_proposal/v2-diff`` schema.
- :mod:`declared_edits` -- checks and hashes a node's 3-5 representative edits
  before any measurement (``scripts/trackg_declare_edits.py``).
- :mod:`smoke` -- a stub-manager smoke campaign for a ported node (G.1 step 4).
"""
