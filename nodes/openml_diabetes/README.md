# OpenML template node

A small binary-classification node on an OpenML dataset, parameterised by
`dataset.json` (node name, dataset ID, dataset name, target column). The frozen
sklearn pipeline and split code live in `benchmark.py`; only bounded
`config.yaml` hyperparameters are editable.

Seeds: `AR_TRAIN_SEED` (model) and `AR_SPLIT_SEED` (train/val/test split) are
injected by the worker; both default to 42.

Created for R1 batch 2 (L8). ENGINEERING_ONLY unless a campaign is authorised.
