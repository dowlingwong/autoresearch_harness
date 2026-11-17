# Node mlgym_houseprice — origin, licence and changes

**Origin.** MLGym (https://github.com/facebookresearch/MLGym, Copyright (c) Meta Platforms, Inc. and
affiliates), task `regressionKaggleHousePrice`, commit `9d40c1b5035202018cd7091fb4e83a9c68b377c0`.

**Licence.** MLGym is released under the Creative Commons Attribution-NonCommercial 4.0 International
licence (CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/). The material of this node that is
taken or adapted from MLGym is under the same licence: it may be used and shared for non-commercial
purposes with attribution, and changes must be indicated (below). The files written by the port (Track G,
Paper B, R1) are listed below.

**Changes made by the port.**
- `train.py` (editable) is adapted from MLGym's `baseline.py`: the same preprocessing and Ridge model,
  returned as an unfitted scikit-learn pipeline; loading, splitting and scoring moved to the frozen wrapper.
- Written by the port: `run_fixed_budget.py` (MLGym's `_helpers/_generate_split.py` procedure with the
  validation split drawn by the split seed; RMSE on validation and test), `ar_data.py`, `prepare_data.py`
  (fetches MLGym's `_helpers/train.csv` at the commit above and checks its sha256 and md5). MLGym's
  `evaluate.py` is replaced by the frozen test-RMSE line.
- The table is MLGym's copy of the Kaggle "House Prices" competition training data (Ames housing data,
  De Cock 2011); Kaggle's competition terms apply to the data.
