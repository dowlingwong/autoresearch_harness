# Node autoresearch_fixedtokens — origin, licence and changes

**Origin.** autoresearch by Andrej Karpathy (https://github.com/karpathy/autoresearch, MIT licence), in the
cross-platform fork miolini/autoresearch-macos as imported into this harness (`nodes/autoresearch-macos`,
imported at harness commit `fb405e1`, platform check changed at `d075bb3`; read at `717ad835`).

**Files.**
- `prepare.py` (frozen): the harness's `nodes/autoresearch-macos/prepare.py`, byte-identical. It holds the
  data download, the tokenizer training, the packing data loader and the metric `evaluate_bpb`.
- `train.py` (editable): the harness's `nodes/autoresearch-macos/train.py` with a fixed training-token budget
  in place of the 5-minute wall-clock budget; every change is listed in its docstring and marked
  "Track G port" in the code.
- Written by the port (Track G, Paper B, R1): `run_fixed_budget.py` (runs train.py, enforces the token budget,
  evaluates `val_bpb`), `ar_budget.py` (the budgeted training loader), `ar_data.py` (hashed data manifest),
  `prepare_data.py` (runs prepare.py's download and tokenizer training into a directory outside the node).

**Data.** The text shards are fetched by prepare.py from `karpathy/climbmix-400b-shuffle` on Hugging Face;
their own terms apply. They are not part of the node.
