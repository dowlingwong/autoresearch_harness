# Node gc7rc_etth1 — licences and provenance

Frozen. Required by `TRACKG_PORT_CHECKLIST.md` section F and `R1-TRACKG-LICENCES`. Licence statements are
recorded as the U1 audit cites them (`docs/r1/reports/trackg_g0_20261117/U1_LICENCE_AVAILABILITY_20261117.md`
§5 and §7, on the paper repository's `ops-tasl-20261116` branch, observed 17 Nov 2026), with the 1GC-7RC
paper's own Appendix E. This is an engineering record, not a legal opinion.

## Code

| Item | Value |
|---|---|
| Upstream suite | 1GC-7RC (arXiv:2605.17046v2). The paper names `https://github.com/Strolchii/1GC-7RC-Benchmark`, which returned **404** when U1 checked it on 17 Nov 2026; **no suite code or licence exists to copy**, and no suite asset is used. |
| Port code | Written by the port (Track G, Paper B, R1), under the repository's MIT licence: `run_fixed_budget.py` (the frozen parent), `ar_worker.py` (the frozen worker entry), `ar_parent.py`, `ar_isolated.py`, `ar_fixed_loop.py`, `ar_data.py`, `prepare_data.py`, `requirements.txt`, this file, `README.md`, and the editable `train.py` (a DLinear baseline implemented from the published DLinear paper, Zeng et al. 2023, for the task the 1GC-7RC paper describes — not suite code). |
| Task semantics | The 1GC-7RC paper, Task 6: multivariate 96-step-ahead forecasting on ETTh1 (7 features), 336 look-back steps, standard LTSF 12/4/4-month split, train-normalised, metric MSE, DLinear baseline. "Implemented by the port from the paper's description; not the suite's assets" is the disclosure for the appendix. |

## Data (never redistributed)

| Item | Value |
|---|---|
| Source | ETDataset (Zhou et al., Informer, AAAI 2021), `https://github.com/zhouhaoyi/ETDataset`, pinned commit `1d16c8f4f943005d613b5bc962e9eeb06058cf07` (ETDataset HEAD recorded by the U1 audit, 17 Nov 2026), file `ETT-small/ETTh1.csv` |
| URL downloaded | `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv` |
| Hash | sha256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`, 2,589,657 bytes; pinned in `prepare_data.py` and checked again by the frozen parent against the data manifest. Every run checks the derived `data/series.npy` hash in `data/data_manifest.json` |
| Licence (three conflicting claims, recorded by U1 §5) | (1) GitHub API reports `other`/`NOASSERTION` and the ETDataset README only asks for the Informer citation; (2) the Hugging Face mirror `thuml/Time-Series-Library` card declares `cc-by-4.0`; (3) OpenDataLab's ETT page (`https://openxlab.org.cn/datasets/OpenDataLab/ETT`, snapshot 17 Nov 2026) claims **CC BY-ND 4.0**; the 1GC-7RC paper's Appendix E also states **CC BY-ND 4.0** |
| Governing licence used | **CC BY-ND 4.0**, the most restrictive of the claims (per `R1-TRACKG-LICENCES`: "ETTh1 licence conflict: use the most restrictive"). Consequences enforced by this node: **the data are not redistributed** (the branch commits only the download script and the hashes; the raw CSV is downloaded at setup and kept outside the repository), the source is attributed (Informer/ETDataset), and **the data are not modified as a derived dataset that is shared** — the node derives the standardised arrays privately at setup for ENGINEERING_ONLY research use, from the unmodified source file whose sha256 is recorded |
| Citation / attribution | Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., Zhang, W. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI 2021. Dataset: ETDataset, https://github.com/zhouhaoyi/ETDataset |

The node and its branch contain no task data: `prepare_data.py` downloads the pinned CSV at setup time into the
data root outside the repository, checks its sha256, derives `series.npy`, records both in the hashed data
manifest and links the directory into the node as `data/`. The port's tests use synthetic random data only.
