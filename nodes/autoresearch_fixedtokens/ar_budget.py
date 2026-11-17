"""Frozen training-token budget of the Track G node autoresearch_fixedtokens.

FROZEN.  The wrapper (run_fixed_budget.py) sets ``TRAIN_TOKENS`` from
``--tokens`` before the editable train.py imports it, and replaces
``prepare.make_dataloader`` by :func:`budgeted`, which passes the validation
loader through unchanged and stops the training loader after exactly
``TRAIN_TOKENS`` tokens: a request beyond the budget raises
:class:`BudgetExhausted` (a BaseException, so ``except Exception`` in edited
code does not hide it) and the run fails.  After train.py returns, the
wrapper checks that the budget was used in full (:func:`served`).
"""
TRAIN_TOKENS = 0  # set by run_fixed_budget.py before train.py is imported
_served = 0


class BudgetExhausted(BaseException):
    """A training batch was requested after the token budget was used up."""


def served() -> int:
    return _served


def budgeted(make_dataloader):
    def make(tokenizer, B, T, split, *args, **kwargs):
        loader = make_dataloader(tokenizer, B, T, split, *args, **kwargs)
        if split != "train":
            return loader

        def train_batches():
            global _served
            for batch in loader:
                if _served + B * T > TRAIN_TOKENS:
                    raise BudgetExhausted(f"batch of {B * T} tokens requested after {_served} of {TRAIN_TOKENS}")
                _served += B * T
                yield batch

        return train_batches()

    return make
