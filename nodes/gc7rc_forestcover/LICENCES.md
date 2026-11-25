# Node gc7rc_forestcover — licences and provenance

Frozen. Required by `TRACKG_PORT_CHECKLIST.md` section F. Licence statements are recorded as the U1 audit
cites them (`docs/r1/reports/trackg_g0_20261117/U1_LICENCE_AVAILABILITY_20261117.md` §3 and §7, on the paper
repository's `ops-tasl-20261116` branch, observed 17 Nov 2026). This is an engineering record, not a legal
opinion.

**Origin.** The task is the 1GC-7RC paper's **Task 5** (tabular prediction): seven-class classification on
**Forest Cover Type** (UCI Machine Learning Repository, dataset 31 "Covertype"; DOI
10.24432/C50K5N; dataset page <https://archive.ics.uci.edu/dataset/31/covertype>). The 1GC-7RC suite
repository named on the paper (`https://github.com/Strolchii/1GC-7RC-Benchmark`) returned **404** in the U1
audit (17 Nov 2026) and again on 25 Nov 2026, so there is no suite code, evaluator, baseline script or suite
licence to port. This node therefore implements the paper's **stated public dataset/task semantics**; the
baseline MLP is written by the port from the paper's description (disclosed in the README and the kit report).

**Licence.** The UCI dataset page states **Creative Commons Attribution 4.0 International (CC BY 4.0,
<https://creativecommons.org/licenses/by/4.0/>)** for Covertype (U1 §3 and §7). Attribution: "Covertype
dataset, UCI Machine Learning Repository (Blackard, J. & Dean, D., 1998), DOI 10.24432/C50K5N, CC BY 4.0".
The paper's own Appendix E also lists Forest Cover Type under CC BY 4.0. This node's own files are the
repository's (MIT); no suite-derived material is used.

**Data: downloaded, never redistributed.** `prepare_data.py` downloads the dataset page's canonical zip
`https://archive.ics.uci.edu/static/public/31/covertype.zip` in the frozen setup step, checks the pinned
sha256, extracts the pinned `covtype.data.gz` member, parses the 581,012 x 55 table and writes the derived
`X.npy` / `y.npy` under `$AR_TRACKG_DATA_ROOT` (outside the repository). The node, the tests and the branch
hold no task data; the raw zip is kept only under the data root's setup-only `raw/` subdirectory, which no run
reads. The legacy direct URL of the same file
(`https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz`) returned the same
11,240,707 bytes with the same sha256 on 25 Nov 2026.

| Item | Value |
|---|---|
| Source | UCI dataset 31, `https://archive.ics.uci.edu/static/public/31/covertype.zip` (HTTP 200 on 25 Nov 2026) |
| Zip sha256 / size | `89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d` / 11,251,206 B |
| Member `covtype.data.gz` sha256 / size | `614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771` / 11,240,707 B |
| Table | 581,012 rows x 55 columns: 54 features (10 continuous + 4 wilderness one-hot + 40 soil-type one-hot) and the class label 1-7 (mapped to 0-6 in `y.npy`) |
| Class counts | 211,840 / 283,301 / 35,754 / 2,747 / 9,493 / 17,367 / 20,510 (classes 1-7) |
| Redistribution | none: the branch holds no data; the tests use synthetic arrays only; setup downloads the bytes at deployment time and verifies the pinned sha256 |

**Changes made by the port.** The task definition (dataset, 54 features, seven classes, stratified 80/10/10
split, StandardScaler normalisation, accuracy metric, batch 4,096) is the paper's. The baseline (3-layer MLP
128-256-128 with BatchNorm, ReLU, dropout 0.3; Adam lr 1e-3; cross-entropy), the fixed-step training loop, the
split, the standardisation and the accuracy computation are the port's own implementation, not suite assets.
No container is used (native port): no image digest applies.
