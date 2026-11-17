"""Frozen training-token budget of the Track G node autoresearch_fixedtokens.

FROZEN.  The wrapper (run_fixed_budget.py) replaces ``prepare.make_dataloader``
by the loader factory that :func:`budgeted` returns, before the editable
train.py runs.  That factory

* stops the training loader after exactly ``tokens`` tokens: a request beyond
  the budget raises :class:`BudgetExhausted`;
* refuses the validation split while train.py runs: a request raises
  :class:`ValidationRequested`, because the validation shard is the only metric
  data and must not become an unbudgeted training source.  The wrapper opens it
  (``open_validation``) only after train.py has returned, for
  ``prepare.evaluate_bpb``.

Both exceptions are BaseExceptions, so ``except Exception`` in edited code does
not hide them, and the wrapper turns each into its own non-zero exit.  The
token counter lives in a closure, not in a module global that train.py could
assign; the wrapper keeps the ``served``, ``violations`` and ``open_validation``
callables.  Each refusal is also recorded before it is raised, so a train.py
that catches the BaseException still fails the run (``violations``).
``TRAIN_TOKENS`` is informational (train.py reads it to plan its steps).
"""


TRAIN_TOKENS = 0  # set by the wrapper before train.py runs, for train.py to plan its steps; not used to enforce


class BudgetExhausted(BaseException):
    """A training batch was requested after the token budget was used up."""


class ValidationRequested(BaseException):
    """train.py asked for the validation split while it was running."""


def budgeted(make_dataloader, tokens: int):
    """(make, served, violations, open_validation): the budgeted loader factory and its controls."""
    state = {"served": 0, "validation_open": False, "violations": []}

    def make(tokenizer, B, T, split, *args, **kwargs):
        if split != "train":
            if not state["validation_open"]:
                state["violations"].append("validation")
                raise ValidationRequested(f"split {split!r} requested while train.py runs")
            return make_dataloader(tokenizer, B, T, split, *args, **kwargs)
        loader = make_dataloader(tokenizer, B, T, split, *args, **kwargs)

        def train_batches():
            for batch in loader:
                if state["served"] + B * T > tokens:
                    state["violations"].append("budget")
                    raise BudgetExhausted(f"batch of {B * T} tokens requested after {state['served']} of {tokens}")
                state["served"] += B * T
                yield batch

        return train_batches()

    def served() -> int:
        return state["served"]

    def violations() -> list:
        return list(state["violations"])

    def open_validation() -> None:
        state["validation_open"] = True

    return make, served, violations, open_validation
