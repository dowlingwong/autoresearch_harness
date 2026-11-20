# Node autoresearch_fixedtokens — licences and provenance

Frozen. Required by `TRACKG_PORT_CHECKLIST.md` section F. Licence statements are recorded as the U1 audit
cites them (`docs/r1/reports/trackg_g0_20261117/U1_LICENCE_AVAILABILITY_20261117.md` §2 and §7, on the
paper repository's `ops-tasl-20261116` branch, observed 17 Nov 2026). This is an engineering record, not a
legal opinion.

## Code

| Item | Value |
|---|---|
| Upstream | https://github.com/karpathy/autoresearch, HEAD `228791fb499afffb54b46200aca536f79142f117` (U1 §2) |
| Upstream licence | **MIT, declared** in the README's `## License` section; upstream ships **no LICENSE file**, and the GitHub API reports `license: null` (U1 §2) |
| Code actually ported | the harness's `nodes/autoresearch-macos` (fork https://github.com/miolini/autoresearch-macos), imported at harness commit `fb405e1` and changed at `d075bb3` (the platform check, and an `AUTORESEARCH_CACHE_DIR` override of the cache directory in `prepare.py`, which the port relies on), read at `717ad835`; so the frozen `prepare.py` is the harness's modified copy, not the fork's file. No fork commit sha was recorded at import. The fork's README carries the same `## License: MIT` statement; neither upstream nor the fork has a LICENSE file or a copyright line. The fork is not covered by U1. |
| Attribution | "autoresearch by Andrej Karpathy (MIT, as declared in its README), via the miolini/autoresearch-macos fork" |

The licence text, copied here as U1 §2 recommends: the standard MIT licence (https://opensource.org/license/mit),
which the README names. Upstream ships no licence file and names no copyright holder; the bracketed line below
says so instead of inventing one.

```
MIT License

Copyright (c) [no copyright holder named upstream; the README credits Andrej Karpathy]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Baseline differences from upstream karpathy/autoresearch (`228791fb`)

The fork, and so this baseline, differs from upstream's `train.py` and environment in:

- `DEPTH` 8 → 4; `TOTAL_BATCH_SIZE` 2^19 → 2^16; `DEVICE_BATCH_SIZE` 128 → 16; `WINDOW_PATTERN` "SSSL" → "L";
- upstream's fast-fail on `math.isnan(loss) or loss > 100` reduced to `loss > 100` (a NaN loss is then caught by
  the loss pattern or a non-finite `val_bpb`);
- attention through PyTorch SDPA with an explicit mask instead of Flash-Attention-3 kernels;
- torch 2.6.0 (`requirements.txt`) instead of upstream's torch 2.9.1 / cu128.
- smaller code differences that should not change results on CUDA: device selection (the fork also accepts MPS
  and CPU; upstream is CUDA-only); `torch.compile` is applied to the fused optimizer steps when the optimizer is
  built rather than by decorator, and the model is compiled only on CUDA (upstream always compiles it), which can
  change speed, not results; scalar casts of the optimizer's hyperparameters.

Upstream's README recommends exactly this kind of reduction (smaller depth, "L" windows, smaller batches) for
small compute. The fork was chosen because upstream loads its Flash-Attention-3 kernels from the Hugging Face
`kernels` hub at import time (`varunneal/flash-attention-3` on Hopper, `kernels-community/flash-attn3` elsewhere),
which needs hub access during every run, the `kernels` package and the torch 2.9.1 pin; the fork runs on plain
PyTorch.

## Files in this node

- `prepare.py`: the harness copy, byte-identical (a test checks this).
- `train.py` (editable): the harness copy with a fixed training-token budget. Every change is listed in its
  docstring and marked "Track G port" in the code.
- Written by the port (Track G, Paper B, R1): `run_fixed_budget.py` (the frozen parent), `ar_worker.py` (the frozen
  worker entry), `ar_parent.py`, `ar_isolated.py`, `ar_budget.py`, `ar_data.py`,
  `prepare_data.py`, `requirements.txt`, this file.

## Data (never redistributed)

| Item | Value |
|---|---|
| Source | https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle; the first 10 training shards and validation shard 06542 (about 1.0 GB, U1 §2) |
| Licence | **MIT, declared on the dataset card** (`license: mit`); the corpus provenance is not documented on the card (U1 §2) |
| Hub revision | U1 observed card revision `915333b`. `prepare_data.py` resolves the current revision, STOPs unless it starts with `915333b` (or the revision passed with `--accept-revision`, after the author accepts it), and downloads the shards from that exact revision |
| Hash | every file's sha256 is written to `data/data_manifest.json` at setup and checked by every run; the census records the manifest's sha256 |
| Tokenizer | trained at setup by `prepare.train_tokenizer` on the downloaded training shards; hashed in the same manifest |

The node and its branch contain no climbmix data. The port's tests use synthetic random-word shards.
