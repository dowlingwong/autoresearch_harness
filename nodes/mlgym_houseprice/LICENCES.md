# Node mlgym_houseprice — licences and provenance

Frozen. Required by `TRACKG_PORT_CHECKLIST.md` section F. Licence statements are recorded as the U1 audit
cites them (`docs/r1/reports/trackg_g0_20261117/U1_LICENCE_AVAILABILITY_20261117.md` §3 and §7, on the
paper repository's `ops-tasl-20261116` branch, observed 17 Nov 2026). This is an engineering record, not a
legal opinion.

**Origin.** MLGym (https://github.com/facebookresearch/MLGym, Copyright (c) Meta Platforms, Inc. and
affiliates; HEAD at U1 §3 and at the port), task `regressionKaggleHousePrice`, commit `9d40c1b5035202018cd7091fb4e83a9c68b377c0`.

**Licence.** Per the repository's `LICENSE` file (U1 §3), MLGym is released under the Creative Commons Attribution-NonCommercial 4.0 International
licence (CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/). The material of this node that is
taken or adapted from MLGym is under the same licence: it may be used and shared for non-commercial
purposes with attribution, and changes must be indicated (below). The files written by the port (Track G,
Paper B, R1) are listed below. MLGym's README notes that SWE-Agent and Modded-NanoGPT are MIT-licensed and Gymnax and
gymnax-blines are Apache-2.0; CC BY-NC 4.0 grants no patent rights (U1 §3). Attribution: "Adapted from MLGym
(Nathani et al., 2025), CC BY-NC 4.0; modified as listed below".

**Scope within the harness repository.** The harness repository's root `LICENSE` (MIT) does not cover this node's MLGym-derived files: they stay under CC BY-NC 4.0 (parts that MLGym took from gymnax or gymnax-blines under Apache-2.0). Recording this in the root `LICENSE`/`README`, or keeping the MLGym nodes outside the MIT tree, is the author's decision (port report, open point 1).

**Changes made by the port.**
- `train.py` (editable) is adapted from MLGym's `baseline.py`: the same preprocessing and Ridge model,
  returned as an unfitted scikit-learn pipeline; loading, splitting and scoring moved to the frozen wrapper.
- Written by the port: `run_fixed_budget.py` (MLGym's `_helpers/_generate_split.py` procedure with the
  validation split drawn by the split seed; RMSE on validation and test), `ar_data.py`, `prepare_data.py`
  (fetches MLGym's `_helpers/train.csv` at the commit above and checks its sha256 and md5). MLGym's
  `evaluate.py` is replaced by the frozen test-RMSE line.

## Data (never redistributed)

| Item | Value |
|---|---|
| Source | MLGym's bundled `data/regressionKaggleHousePrice/_helpers/train.csv` (460,676 B), fetched by `prepare_data.py` from the pinned MLGym commit on raw.githubusercontent.com, or taken from a local MLGym checkout with `--from-file` |
| Origin and terms | Kaggle competition "House Prices — Advanced Regression Techniques" (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques; Ames housing data, De Cock 2011). Kaggle's competition rules and terms apply; MLGym's CC-BY-NC does not relicense the data. **Ambiguous: no licence is claimed** (U1 §3). U1 inferred the file's identity with the Kaggle training file from its name and columns; it was not checked row by row |
| Hash (pinned) | sha256 `1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41`, md5 `80ccab65fb115cbad143dbbd2bcd5577`, computed from the file at the pinned commit and checked by `prepare_data.py` and by every run |

No container is used (native ports): no image digest applies. The node and its branch contain no task data; the
port's tests use synthetic data.
