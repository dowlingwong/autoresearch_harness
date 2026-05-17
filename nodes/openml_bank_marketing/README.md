# OpenML bank-marketing Node

Public tabular binary-classification node using OpenML dataset `1461`
(`bank-marketing`).

Only `config.yaml` is editable by governed campaigns. The dataset ID, target handling,
split seed, preprocessing code, training script, and metric logging are frozen.

Run:

```bash
python3 train.py --config config.yaml
```
