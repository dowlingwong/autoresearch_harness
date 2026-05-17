# Reference Fact Check

Date: 2026-05-16  
Scope: `A-Governed-Harness-for-Auditable-LLM-Driven-ML-Experimentation/references.bib`

All 26 bib entries audited. 22 are cited in the paper; 4 are in the bib but uncited.
Entries are listed in **first-appearance citation order**, uncited entries appended at the end.

---

## Master Table

| # | Bib key | Type | URL ✓ | DOI ✓ | Risk | Issue | Recommended action |
|---|---|---|---|---|---|---|---|
| 1 | `trivedy2026anatomy` | online | ✓ | N/A | **Low** | None. `langchain.com/blog/the-anatomy-of-an-agent-harness` resolves. | Keep as-is. |
| 2 | `jiang2025aide` | preprint | ✓ arXiv | ✓ `10.48550/arXiv.2502.13138` | **Low** | None. Full author list present. | Keep as-is. |
| 3 | `huang2024mlagentbench` | Inproceedings | ✓ PMLR | ✗ missing | **Low** | PMLR entry confirmed. No DOI field in entry; PMLR DOIs exist via DBLP. | Optional: add `doi = {10.5555/3692070.3692905}` (verify via DBLP). Non-blocking. |
| 4 | `chan2024mlebench` | preprint | ✓ arXiv | ✓ `10.48550/arXiv.2410.07095` | **Low** | None. | Keep as-is. |
| 5 | `zou2025fmlbench` | preprint | ✓ arXiv | ✓ `10.48550/arXiv.2510.10472` | **Low** | None. | Keep as-is. |
| 6 | `nathani2025mlgym` | preprint | ✓ arXiv | ✓ `10.48550/arXiv.2502.14499` | **Low** | None. | Keep as-is. |
| 7 | `birk2026sharp` | preprint | ✓ arXiv | ✓ `10.48550/arXiv.2604.18752` | **Low** | None. Author list verified. | Keep as-is. |
| 8 | `huggingface2026mlintern` | ArtifactSoftware | ✓ GitHub | N/A | **Low** | Named preferred-citation authors present (not org alias). | Keep as-is. |
| 9 | `dzhng2026deepresearch` | ArtifactSoftware | ✓ GitHub | N/A | **Low** | ~~Author was raw GitHub handle `{dzhng}`.~~ Fixed: real name is **David Zhang** (`github.com/dzhng`). URL confirmed public. | ✅ Fixed. |
| 10 | `burtenshaw2026multiautoresearch` | ArtifactSoftware | ✓ GitHub | N/A | **Low** | ~~Author was raw GitHub handle `{burtenshaw}`.~~ Fixed: real name is **Ben Burtenshaw** (HuggingFace profile: `huggingface.co/burtenshaw`, LinkedIn: Community Education, HuggingFace). | ✅ Fixed. |
| 11 | `nousresearch2026hermesagent` | ArtifactSoftware | ✓ GitHub | N/A | **Low** | `github.com/NousResearch/hermes-agent` confirmed public and active (latest release v0.13.0, May 7 2026). | ✅ Verified; no change needed. |
| 12 | `bockeler2026coders` | online | ✓ | N/A | **Low** | ~~URL was `…/exploring-gen-ai/harness-engineering.html` — wrong path.~~ Fixed to `martinfowler.com/articles/harness-engineering.html`. | ✅ Fixed. |
| 13 | `hashimoto2026journey` | online | ✓ | N/A | **Low** | `/writing/my-ai-adoption-journey` verified. | Keep as-is. |
| 14 | `openai2026codex` | online | ✓ | N/A | **Low** | `openai.com/index/harness-engineering/` verified. Author Ryan Lopopolo correct. | Keep as-is. |
| 15 | `anthropic2025longrunninga` | online | ✓ | N/A | **Low** | `anthropic.com/engineering/effective-harnesses-for-long-running-agents` verified. Author Justin Young correct. | Keep as-is. |
| 16 | `mlflow` | ArtifactSoftware | ✓ | N/A | **Low** | `mlflow.org` stable. Full 2018 USENIX OpML author list present. | Keep as-is. |
| 17 | `wandb` | ArtifactSoftware | ✓ | N/A | **Low** | `wandb.ai` stable. | Keep as-is. |
| 18 | `hutter2019automl` | book | N/A | ✓ `10.1007/978-3-030-05318-5` | **Low** | Springer book with valid DOI. | Keep as-is. |
| 19 | `elsken2019nas` | article | ✓ JMLR | ✗ missing | **Low** | JMLR vol 20 no 55 confirmed; URL stable. JMLR DOIs exist but are often omitted in practice. | Optional: add `doi = {10.5555/3322706.3322729}` (verify via JMLR). Non-blocking. |
| 20 | `deng2023mind2web` | Inproceedings | ✓ arXiv | ✗ missing | **Low** | Published NeurIPS 2023 (pp. 28091–28114). Entry uses arXiv URL for a proceedings paper. No DOI listed. | Optional: add `doi = {10.48550/arXiv.2306.06070}` or NeurIPS proceedings DOI. Non-blocking. |
| 21 | `anthropic2026longrunningb` | online | ✓ | N/A | **Low** | `…/harness-design-long-running-apps` verified. Author Prithvi Rajasekaran correct. | Keep as-is. |
| 22 | `trivedy2026betterharness` | online | ✓ | N/A | **Low** | Full slug `…better-harness-a-recipe-for-harness-hill-climbing-with-evals` verified. | Keep as-is. |

---

## Uncited Entries — Removed

All four orphan entries were **removed** from the bib to keep the bibliography clean for submission.

| # | Bib key | Action | Reason |
|---|---|---|---|
| 23 | `lu2024aiscientist` | ✅ Removed | Valid entry but uncited; no prose hook in §2 to attach it to without rewriting. |
| 24 | `karpathy2026autoresearch` | ✅ Removed | Uncited; also had wrong `month = apr` (correct: March 7 2026). |
| 25 | `langgraph` | ✅ Removed | Uncited despite §4 describing `langgraph_manager`; prose names the tool without a formal citation. |
| 26 | `langchain` | ✅ Removed | Uncited; same rationale as `langgraph`. |

---

## Priority Fix List — Status

| Fix | Status |
|---|---|
| `bockeler2026coders` URL → `martinfowler.com/articles/harness-engineering.html` | ✅ Done |
| `dzhng2026deepresearch` author `{dzhng}` → `Zhang, David` | ✅ Done |
| `nousresearch2026hermesagent` URL verified | ✅ Confirmed public |
| 4 uncited orphan entries removed (`lu2024aiscientist`, `karpathy2026autoresearch`, `langgraph`, `langchain`) | ✅ Done |
| `burtenshaw2026multiautoresearch` author `{burtenshaw}` → `Burtenshaw, Ben` | ✅ Done (2026-05-16, verified via HuggingFace/LinkedIn) |

**Remaining optional (non-blocking, strict ACM formatting):**

- Add DOI to `huang2024mlagentbench` (row 3), `elsken2019nas` (row 19), `deng2023mind2web` (row 20).

---

## Previously Fixed (confirmed resolved in this bib)

| Bib key | Old problem | Status |
|---|---|---|
| `openai2026codex` | Wrong author (Alessandro Lopopolo) + dead URL `/research/codex-harness-engineering` | ✓ Ryan Lopopolo + `openai.com/index/harness-engineering/` |
| `anthropic2025longrunninga` | Wrong author (Amanda Young) + wrong slug | ✓ Justin Young + correct slug |
| `anthropic2026longrunningb` | Wrong author (Anu Rajasekaran) + truncated slug | ✓ Prithvi Rajasekaran + `…/harness-design-long-running-apps` |
| `hashimoto2026journey` | Wrong slug `/writing/ai-journey` | ✓ `/writing/my-ai-adoption-journey` |
| `trivedy2026betterharness` | Truncated slug `/better-harness/` | ✓ Full slug |
| `jiang2025aide` | `and others` + wrong `primaryClass` | ✓ Full author list + `cs.AI` |
| `huggingface2026mlintern` | Org-author alias only | ✓ Named preferred-citation authors |
| `bockeler2026firstthoughts` | Duplicate entry, Thoughtworks URL unverified | ✓ Merged into `bockeler2026coders`; Thoughtworks entries removed |
| `bockeler2026coders` URL | Wrong path `exploring-gen-ai/harness-engineering.html` | ✓ Fixed to `martinfowler.com/articles/harness-engineering.html` |
| `dzhng2026deepresearch` author | Raw handle `{dzhng}` | ✓ Fixed to `Zhang, David` (verified via github.com/dzhng) |
| `lu2024aiscientist`, `karpathy2026autoresearch`, `langgraph`, `langchain` | Uncited orphan entries | ✓ Removed from bib |
| `burtenshaw2026multiautoresearch` author | Raw handle `{burtenshaw}` | ✓ Fixed to `Burtenshaw, Ben` (verified via HuggingFace profile) |
