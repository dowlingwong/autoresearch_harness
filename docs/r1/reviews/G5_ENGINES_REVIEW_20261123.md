Reviewed base: G5-7 on `c025f89a`, G5-AR on `2d09f9a9`; accepted at `94a41a7f` / `0aa27e31`, confirmed at `40396dc684980308f321a94280311dfc166a75a2` / `134b03994250367b2ee5ca2cd4940c929e1cb9c2`

# Independent review record — the G.5 harness engines (`trackg/audit_g5.py`: campaign and audit) at both G.5 harness commits

**Unit.** In the harness repository, byte-identical G.5 changes on `mac-trackg-g5-7-20261123` (on `c025f89a`) and `mac-trackg-g5-ar-20261123` (on `2d09f9a9`): new `src/autoresearch/trackg/audit_g5.py` (`ENGINE_CONTRACT = "g5_tasl_engine_contract/1"`; campaign: instance seeds 5101–5112 (5100 + k per wave), the G.4 v2-diff path with ledgers at `instance_<k>_<mode>/campaign_ledger.jsonl`, `campaign_record.json`, `check_effective` on every run, `cache_prompt: false`, truncation at the completion cap invalid (`TruncationInvalidManager`), `code_state_sha256` at the top level of every ledger record with a code state, the deadline checked before every instance, manager call and run attempt (`CampaignGuard`); audit: units re-materialised and hashed against the chunk, seeds from the K block (7001–7010 or 7001–7003, 7006–7008), `rows.jsonl` and `chunk_record.json`, `ok` rows with a finite `outcome.value`, isolation block and `executed`, infrastructure after 2 reruns; attribution: an OOM kill at the scope's memory limit divergent, a parent fault signal or a default infrastructure log pattern after `AR_DEVICE` infrastructure on the first attempt and divergent only on recurrence at the same state and seed, in audits and campaigns; a catch-all and a structured `stop_kind` in both records), `tests/trackg/test_trackg_audit_g5.py` (63, including an end-to-end run piping real engine output through the paper's `campaign_file`, `campaign_diagnostics`, `tar_members`, `audit_plan`, `audit_chunk_file`, `read_rows` and `reduce_rows`), one parameter `runner_factory` in `campaign_v2_diff.py` (default unchanged), and the report `docs/r1/reports/trackg_g5_20261123/G5_ENGINES_20261123.md`. `pilot_g4.py`, `noise_floor.py` and `manager_v2_diff.py` are unchanged. The paper's `g5_pin_check`, `g5_job.check_engines` and `g5_tasl.measurement_path_check` pass at both tips. Authority: the user's queue of 23 Nov 2026 (14:02), item 2; the paper-side contract of the G.5 code (READY 91).

**Process.** One independent reviewer, read-only, synthetic nodes and fakes only, isolation waived in this container (so smoke items on TASL-2 are mandatory). Round 1 found one major (the campaign engine checked the deadline only between instances, so the job's hard kill could leave the edited code running on the GPU unsupervised) and six minors; round 2 accepted with three non-blocking points (the after-`AR_DEVICE` rule without a recurrence test; a note line for the campaign attribution change; disclosure of a two-attempt persistent host fault), applied and confirmed.

| Round | Reviewed | Verdict |
|---|---|---|
| 1 | `3b94504b / 4ed74a94` | NOT ACCEPTED (M1; minors m1–m6) |
| 2 | `94a41a7f / 0aa27e31` | ACCEPTED (three non-blocking points) |
| 3 | `40396dc6 / 134b0399` | ACCEPTED (confirmation) |

**Open items (report §6, §7).** Paper side (a follow-up unit on READY 91's code): `g5_job.read_rows` also requires `log_intact`, no `parent_traced_by`, no `parent_dumpable_at_accept`, no `gate_refusal` (else STOP 2); `stop_kind = deadline` maps to node-class STOP 9 in `audit_diagnostics` and `campaign_diagnostics` before the missing-rows checks; on `TimeoutExpired`, `cgroup.kill` the scope's `ar_*` child cgroups before tarring; `-p OOMPolicy=continue` in `t1_tasl_node.sh`; `engine_contract()["truncation_invalid_in"]` names `audit_g5.TruncationInvalidManager`. For the author's signature: the `campaign_attribution` note line (§7 item 6). TASL-2 smoke items 1–15 before any EVIDENCE job (binary outputs left unread; edit-caused OOM and crash classified divergent; rebuilt hashes equal to the ledger's; `outcome.value` on every ok row; isolation effective through the real gate; one forced infrastructure row; the top-level code state on a not-run duplicate; timing; systemd OOM handling (`OOMPolicy`, `memory.oom.group` = 0, systemd-oomd); no `ar_*` processes after a deadline STOP; the CPU-only node; `usage.completion_tokens` on a cut-off reply; disk use of `out/`; one job through the real `g5_job.run_job`). Disclosed residuals: an OOM kill landing on the engine leaves no record; the census time is capped at 300 s; a host fault persisting across two consecutive attempts yields a divergent row carrying its attribution.

## Round 1 (`3b94504b / 4ed74a94`)

> Reviewed base: 3b94504b6ccf1949636c40d3a6e0ff3d25cf45e7 (G5-7, `mac-trackg-g5-7-20261123`) and 4ed74a94c0afcbde4b9f856000ca3fd6bb3bd00e (G5-AR, `mac-trackg-g5-ar-20261123`)
>
> ## Summary
>
> The unit is careful work. It matches the paper's contract on almost every item.
> - **Byte identity and pins hold.**
> - **Every paper-side reader I ran accepts what the engines produce.**
> - **The existing callers are unchanged.** G.4's behaviour stays the same under the new `runner_factory`, and `pilot_g4` and `noise_floor` keep their guards.
> - **The OOM rule is sound within its premises.**
>
> I found one Major defect: the campaign engine can run past the job's deadline in the middle of an instance. On TASL-2 the job then kills the engine, and the isolated run it was supervising keeps going on the GPU with no timeout. I also found six minor points. The author's four open decisions are either author decisions or paper-side follow-ups; none is a defect here.
>
> ## Checks I ran (read-only, scratch moved to /tmp/hg5rev/_to_delete/)
>
> **1. Byte identity and pins**
> - `git diff c025f89a 3b94504b` and `git diff 2d09f9a9 4ed74a94` are byte-identical, blob ids included.
> - The file hashes match the report: `audit_g5.py` a25ddbbc…, `campaign_v2_diff.py` 904e498b…, the test file bfad998b….
> - The pin delta `c025f89a..2d09f9a9` is identical to `3b94504b..4ed74a94` (10 paths), so the pin files are untouched in both families.
> - `g5_pin_check.py /tmp/pb/harness 3b94504b… 4ed74a94…` exits 0. Its `g5_changes` are the report, `audit_g5.py`, `campaign_v2_diff.py` and the test file.
>
> **2. `check_engines` and `measurement_path_check`**
> - `g5_job.check_engines` (real `git grep`) passes at both tips, from `/tmp/pb/harness` and from both worktrees.
> - The negative control STOPs at `c025f89a` and at `2d09f9a9`.
> - `g5_tasl.measurement_path_check` passes for both families:
>   - 308 measurement-path files are unchanged;
>   - the added files are only the report, `audit_g5.py` and the test file;
>   - `engine_files_in_closure` is `[]`, and the import closure is unchanged (15 modules).
> - The cross-family pairing (`c025f89a` against G5-AR) STOPs, as it should.
>
> **3. Tests** (one file at a time, both worktrees give identical results; both worktrees stayed clean)
>
> | File | Result |
> |---|---|
> | `test_trackg_audit_g5.py` | 53 passed |
> | `test_trackg_campaign_v2_diff.py` | 16 passed |
> | `test_trackg_pilot_g4.py` | 31 passed |
> | `test_trackg_noise_floor.py` | 33 passed |
> | `test_trackg_isolation.py` | 28 passed |
>
> **4. Extra probes of my own**
> - **Paper-made input files.** `g5_job.campaign_file` (waves 1 and 2) and `audit_chunk_file` (K = 10 and K = 6), built for all 8 nodes from the real `g5_tasl_spec.json`, all pass the engine's `validate_campaign` / `validate_chunk`.
> - **Diagnostics on a real campaign.** `g5_job.campaign_diagnostics` on a real campaign output gives `stop: None` and passes `check_diagnostics`. After `tar -cf out.tar out`, `LEDGER_MEMBER` finds exactly the 4 ledgers among 158 files.
> - **Probes for findings.** One probe shows the deadline overrun (M1), another the gate-refusal classification (m1).
>
> ## Contract fidelity, item by item: met
>
> - **Schemas and marker.** Present.
> - **Members:**
>   - campaign ledgers at `instance_<k>_<mode>/campaign_ledger.jsonl`, matching `history|none`;
>   - `out/audit/rows.jsonl` via `--out out/audit`;
>   - `out/chunk.json` is never touched by the engine.
> - **Records.** `campaign_record.json` and `chunk_record.json` both carry `stop`.
> - **Rows.** `ok` rows carry a finite float `outcome.value`. The kinds are `ok`, `divergent` and `infrastructure_failure`. The isolation block and `executed` are present.
> - **Infrastructure reruns.** An infrastructure failure becomes one row after 2 reruns. A `RunnerError` gives `executed: false` and no isolation block.
> - **Top-level `code_state_sha256` on not-run steps.** Lifted from the diff provenance. For an inert diff this is the incumbent's normalised state, so it agrees with `g5_assemble`'s `parent_state` fallback.
> - **`cache_prompt: false`.** Enforced, and recorded in every request.
> - **Stop text.** Names a cause and an index only; the job maps it to a fixed reason anyway.
> - **Seed guards:**
>   - instance seeds 5101–5112, with 5100 + k per wave;
>   - audit seeds from the K block: K = 10 is 7001–7010; K = 6 is 7001–7003 and 7006–7008;
>   - `pilot_g4` and `noise_floor` are byte-unchanged and keep their guards.
> - **No bypass.** I found no path for G.5 to skip isolation or use other seeds:
>   - the CLI builds `isolation_waivers={}`, refuses unisolated nodes, and uses the default runner and gate;
>   - the runner itself refuses root, ptrace_scope < 1, a missing cgroup and a missing sandbox.
> - **`runner_factory`.** Only the call site changes, with the default `WorkingCopyRunner`, so G.4 is byte-identical in behaviour.
> - **Truncation in `audit_g5.py` rather than `manager_v2_diff.py`.** Acceptable in substance: same rule, `completion_tokens >= max_tokens`, which is the job's counting rule. See decision 2.
>
> ## Findings
>
> ### B (blocking)
> None.
>
> ### M1. The campaign engine can overrun `deadline_epoch` in the middle of an instance, leaving an unsupervised run on the GPU
> - **Where:** `audit_g5.run_campaign`. The deadline is checked only before each instance starts. `G5CampaignRunner` and the manager never see it.
> - **Problem:** On TASL-2 the deadline is a hard kill. `g5_job` gives the engine `subprocess.run(..., timeout=remaining)`, so at the deadline it SIGKILLs the engine process only. The isolated run that engine was supervising is in its own session and run cgroup:
>   - nothing sets PDEATHSIG;
>   - the runner's 3 × census timeout dies with the runner.
>
>   So the edited code keeps running on the GPU after the job releases the lease. It runs until its current call ends, which is unbounded for a looping edit.
> - **The author saw this hazard.** Open question 3 says that without the audit guard "the job's subprocess timeout would fire instead… it could leave an orphaned run on the GPU". The audit engine got a per-run guard; the campaign did not.
> - **Probe:** a real wave-1 campaign with the deadline 2 s ahead. Instance 1 ran all 4 runs and finished 7.3 s past the deadline. Only instance 2 was refused.
> - **Failure scenario:** a slow wave-2 campaign: 8 instances × 21 calls.
>   - *ogbn_arxiv:* a timeout is 3 × 21.8 s, while the budget is (5.65 + 60) × 1.5 ≈ 98 s per call.
>   - *autoresearch_fixedtokens:* a timeout step costs about 828 s against a budget of about 418 s per call.
>   - *What happens:* the job kills the engine mid-run with node-class STOP 9. Then:
>     - the orphan keeps writing into `out/campaign/...` while `tar` reads it, so the tar can fail and `out/` stays in clear;
>     - the next job STOPs on the foreign-process check;
>     - the operator has to clean up by hand.
> - **Fix (small):**
>   - Check the deadline before every run attempt. `G5CampaignRunner.run` can raise `G5EngineStop` when `clock() + 3 × census + 60 > deadline_epoch`.
>   - Check it before every manager call, for example in `TruncationInvalidManager.propose`, against the run budget plus an LLM allowance.
>   - `run_v2_diff_campaign` lets `G5EngineStop` propagate; it catches only `IntegrityError` and `ManagerTransportStop`. `run_campaign` already records the stop.
>   - Add a clock-driven test.
>   - **Paper-side defence in depth:** on `TimeoutExpired`, `g5_job` should `cgroup.kill` the `ar_*` children of the scope.
>
> ### Minor
>
> **m1. A gate refusal on `log_intact` is classed as infrastructure, not STOP 2, and report item 13 overstates.**
> - **Where:** `run_audit`'s gate path, against `g5_job.read_rows`, which checks only `effective`, `waivers` and `exceptions`.
> - **Problem:** Suppose the gate refuses with `effective: true` and `log_intact: false` (outcome `divergent log_tampered`). The row is written, and `read_rows` does not raise STOP 2.
> - **Probe:** the job got "STOP: 1 planned audit runs have no row", which is infrastructure class. If the refused run is the chunk's last, the job sees the engine stop instead, also infrastructure.
> - **Failure scenario:** a real log-integrity breach permits a new attempt instead of a node STOP.
> - **Fix:**
>   - Paper side: `read_rows` should also require `log_intact is True`, no `parent_traced_by`, no `parent_dumpable_at_accept`, and no `gate_refusal`. The row already carries all of these.
>   - Harness side: correct item 13 of the report.
>
> **m2. `crash_signal` effectively applies only to the frozen parent.**
> - **Where:** `attribute()`, `CRASH_SIGNALS`.
> - **Problem:** In the isolated design the edited code runs only in the sandboxed worker. The launcher turns a worker's signal death into exit 1, so the parent exits non-zero, which is already divergent `nonzero_exit`. The re-attribution therefore fires only for a fault of the frozen parent. Edits can hardly cause that; a hardware SIGBUS (memory error), a library abort or a harness bug can.
> - **Failure scenario:** a transient host fault becomes a silent divergent label at that seed, where it should be a rerun.
> - **Fix:** class a parent fault signal as infrastructure on its first attempt, and as divergent only if it recurs on the rerun at the same (unit, seed). Or keep the rule and disclose it in the note.
>
> **m3. The default infrastructure log patterns can match the edited code's own output.**
> - **Where:** `outcome.DEFAULT_INFRASTRUCTURE_LOG_PATTERNS`, for example `No space left on device` and `Input/output error`. They match anywhere in the log, including the worker's output. The node-specific patterns are carefully anchored before `AR_DEVICE`; these are not.
> - **Problem:** `attribute()` leaves `infrastructure_log_pattern` as infrastructure.
> - **Failure scenario:** an edit that fills the disk or prints such a line gives infrastructure at every seed → one infrastructure row → STOP 7 → a new attempt → a loop. That is exactly what ruling 1 warns against.
> - **Fix:** treat a default-pattern match after the first `AR_DEVICE` line as divergent. Or record it as an author-signed residual and add a smoke check.
>
> **m4. No catch-all exception handling, and no structured stop code.**
> - **Problem:** `run_audit` and `run_campaign` catch a fixed list of exceptions. Anything else crashes the engine without writing `chunk_record.json` or `campaign_record.json`. The job then sees "no chunk_record.json", which is infrastructure class.
> - **Fix:** add a final `except Exception` that records `stop`. Also add a structured `stop_kind` field (for example `deadline`, `integrity`, `unisolated`, `refused`), so the paper can map stop classes without parsing text (see decision 3).
>
> **m5. The TASL-2 smoke list (report §6) is incomplete.** Add:
> - systemd-oomd, that is `ManagedOOMMemoryPressure` on `user@.service` / `oomctl`. A userspace kill leaves `oom` and `oom_kill` unchanged, so an edit-caused memory-pressure kill would be infrastructure and would loop;
> - the scope's `memory.oom.group` is 0;
> - after a deadline STOP or an engine kill, the scope has no `ar_*` processes left (relates to M1);
> - one CPU-only node (`mlgym_houseprice`): no `--gpu-uuid`, `CUDA_VISIBLE_DEVICES` empty;
> - every llama-server reply carries `usage.completion_tokens`, and it equals 4096 on a cut-off reply (`finish_reason: length`). The truncation rule depends on it;
> - the size of `out/` per campaign wave against free disk. Working copies and snapshots are kept, and the tar briefly doubles the space;
> - one job through the real `g5_job.run_job`, using the `g5` kind of `t1_tasl_node.sh` in a delegated scope.
>
> **m6. Report and test gaps.**
> - **Wording:**
>   - "the scope holds only this engine and the run" is not accurate: the scope also holds `g5_job` and the wrapper.
>   - Open question 2 should note that `Delegate=yes` makes `continue` the scope default (see decision 1).
> - **End-to-end test.** It is valuable: real diff application, real re-materialised hashes, and the paper's `parse_ledger`, `audit_plan`, `chunk_plan`, `audit_chunk_file`, `read_rows`, `reduce_rows` and `campaign_counts` run on real engine output. Its gaps:
>   - It uses the test's own campaign file, not the paper's `campaign_file`.
>   - It does not run `campaign_diagnostics` or `tar_members` on real output. My probes confirm both work; worth adding as tests.
>   - Isolation is waived, and the record's isolation verdict is overwritten after each run. So the sandbox, the cgroup and the `memory.events` attribution are not exercised for real. That is acceptable in this container, but it makes smoke items 2 and 6 mandatory.
>
> ## Rulings on the author's four decisions
>
> 1. **systemd `OOMPolicy` on TASL-2: a paper-side follow-up plus a smoke check, not a defect here.** The [systemd.scope man page](https://www.freedesktop.org/software/systemd/man/253/systemd.scope.html) says scopes inherit `DefaultOOMPolicy=`, but units with `Delegate=` turned on default to `continue`. The wrapper passes `-p Delegate=yes`, so the risk is low. Still:
>    - add `-p OOMPolicy=continue` to `t1_tasl_node.sh` explicitly;
>    - confirm `OOMPolicy`, `memory.oom.group = 0` and systemd-oomd on the host (m5).
> 2. **Where truncation lives: acceptable, with a paper-side wording follow-up.** Keeping `manager_v2_diff.py` and G.4 unchanged is better. But `engine_contract()["truncation_invalid_in"]` goes into every job record via `record["engine"]`. Before the signature, update it to name `audit_g5.TruncationInvalidManager`.
> 3. **Engine deadline STOP as infrastructure: a paper-side follow-up, required before launch.** Keep the guard; it is correct. The 3b-1 review (C5) states that a missed deadline is a node STOP. So `audit_diagnostics` and `campaign_diagnostics` should map an engine deadline stop to node-class STOP 9, before the missing-rows check. The structured `stop_kind` from m4 would make this robust. Once M1 is fixed, the same applies to campaigns.
> 4. **Campaigns keep G.4's classifier for memory-limit kills: the author's decision, to be disclosed and signed.** Ruling 1 was stated for the audit infrastructure bit. In campaigns, `infrastructure_failure_final` is bounded (no loop), and a worker OOM is already `discarded_divergent`. But a parent killed by an edit-caused OOM drops out of the table instead of being labelled, which is a small selection effect. I recommend carrying `attribute()` into `G5CampaignRunner`; it is cheap, and the hook already exists.
>
> ## Memory
> Nothing new worth saving. I did not use memory for this review.
>
> VERDICT: NOT ACCEPTED (M1: the campaign engine checks `deadline_epoch` only between instances, so the job's hard kill can land mid-run and leave the edited code running on the GPU with no supervision; add per-run and per-call deadline checks with a test, then fix m1–m6 or note them)

## Round 2 (`94a41a7f / 0aa27e31`)

> Reviewed base: 94a41a7f8f6ac08b5691a76143afebd18a6c2d77 (G5-7) and 0aa27e31598ae0e3228dbfc4b44f182d2b3b791d (G5-AR). Round 2, on 3b94504b and 4ed74a94.
>
> ## Summary
>
> Round 2 fixes M1 (the campaign deadline) correctly, and I confirmed it by running the code. It also handles m1–m6 and decision 4. I checked the author's summary claims against the code and test runs, and they hold. Three points remain, none blocking:
> - m3's after-`AR_DEVICE` rule turns a one-off host I/O error during training into a divergent label.
> - Decision 4 changes what the paper side tables and counts, so it needs a note line for the author to sign.
> - A persistent host fault can still give a `crash_signal_repeated` row, which should be disclosed.
>
> ## What I checked myself
>
> **Byte identity and pins**
> - `git diff 3b94504b 94a41a7f` and `git diff 4ed74a94 0aa27e31` are byte-identical.
> - The pin delta `c025f89a..2d09f9a9` is identical to `94a41a7f..0aa27e31`.
> - File hashes match the author's summary: `audit_g5.py` c674c53a… and the test file 859d1bdf… in both families. `campaign_v2_diff.py` is unchanged (904e498b…).
>
> **Paper-side checks**
> - `g5_pin_check` exits 0 at both new tips.
> - `check_engines` passes at both tips, from all three checkouts. The `c025f89a` negative control still STOPs.
> - `measurement_path_check` passes for both families: 308 paths unchanged, nothing new added beyond the report, `audit_g5.py` and the test file, `engine_files_in_closure` is `[]`, and the import closure is unchanged (15 modules).
>
> **Tests** (one file at a time, identical in both worktrees; all trees stayed clean)
>
> | File | Result |
> |---|---|
> | `test_trackg_audit_g5.py` | 62 passed |
> | `test_trackg_campaign_v2_diff.py` | 16 passed |
> | `test_trackg_pilot_g4.py` | 31 passed |
> | `test_trackg_noise_floor.py` | 33 passed |
> | `test_trackg_isolation.py` | 28 passed |
>
> **My round-1 deadline probe, re-run on real runs.** Every case finished before the deadline:
> - 2 s of slack: the baseline run is refused (`stop_kind: deadline`).
> - 130 s: the baseline runs, then the step-1 manager call is refused, with 126.7 s still left.
> - 300 s: all 4 instances complete.
>
> ## M1, the deadline: fixed, and the margins are sound
>
> **What the code does now**
> - `CampaignGuard.check_run` runs before every run attempt, reruns included, because `runner_factory` builds a fresh runner for each attempt. A run starts only if 3 × census + 60 s fit before the deadline.
> - `check_call` runs before every manager call and needs 120 s more.
> - The instance check is kept.
> - `G5EngineStop(kind="deadline")` passes through `run_v2_diff_campaign` (it catches only `IntegrityError` and `ManagerTransportStop`) and is recorded.
>
> **Is 60 s enough on top of the timeout?** It covers what happens around the timed part of a run:
> - after a timeout: the 10 s cgroup kill grace, the 10 s drain grace and the final cgroup kill;
> - before the timer starts: the kit copy (the kits are 132–204 KB) and the sandbox probe.
>
> The job's `int(remaining)` subprocess timeout can kill the engine up to 1 s early, which the margin also absorbs.
>
> **Slow nodes**
> - `autoresearch_fixedtokens`: a run needs 828 s left and a call 948 s. The wave deadline is 1800 s plus 1.5 × the budget (about 35,000 s for wave 1), so the guard only bites when a wave really overruns, which is what it is for.
> - `ogbn_arxiv`: a run needs 125 s and a call 245 s, against a wave-1 deadline of about 10,000 s.
>
> **A slow LLM call cannot recreate the orphan problem.** If a call runs past its 120 s allowance, the run check after it still refuses the run. If the job kills the engine during a call, no run is in flight. So the 120 s figure only decides which stop class you get, not safety. A stop in the middle of an instance leaves a ledger without a STOP line. The paper side treats it as a stopped attempt, and §7 item 2 asks it to map `deadline` to node-class STOP 9.
>
> ## m2, the rerun rule: mislabel risk is narrow
>
> `prior_crash` counts only earlier attempts of the same planned (unit, seed) in audits, and the same (state, train seed) in campaigns. So a parent fault becomes divergent only if it hits twice in a row on consecutive attempts.
>
> The remaining risk is a host fault that persists across two consecutive attempts, minutes apart. If it persists longer, it also hits the baseline and parent units, so E5 STOPs the node through `baseline_label_divergent_other`. If it lasts only those two attempts, one row is mislabelled. That is rare and acceptable. The row carries `attribution.crash_signal`, so the T1 report can count such rows; it should disclose them (non-blocking).
>
> ## m3, the `AR_DEVICE` rule: it closes the loop, with one residual
>
> **The marker is in the right place.** In the nodes I read (`mlab_cifar10`, `mlgym_fmnist`, `autoresearch_fixedtokens`), `AR_DEVICE` is printed by frozen worker code before any editable code is imported or run. Edited code cannot fake an earlier `AR_DEVICE` line. Node-specific patterns and `mlgym_houseprice`, which has no `AR_DEVICE` line, stay infrastructure; the houseprice case is disclosed.
>
> **Residual.** A genuine host fault after `AR_DEVICE` becomes divergent at once, with no recurrence test. Examples: an `Input/output error` while `autoresearch_fixedtokens` reads its data inside the editable phase, or disk-full caused by something else. A one-off host fault therefore turns into a silent label.
>
> **Recommendation (non-blocking):** apply m2's rule here too. A first after-`AR_DEVICE` match is infrastructure and rerun; it becomes divergent only on recurrence at the same state and seed. That still cannot loop.
>
> **Nit.** `classify_run` reports only the first default pattern that matches. A different default pattern matching before `AR_DEVICE` would be ignored. Either check that no default pattern matches before `AR_DEVICE`, or accept this edge case.
>
> ## Decision 4: it changes what the paper side reads, but breaks no reader
>
> **No reader breaks.** The ledger keeps its schema. The run record gains `outcome_attribution`, which `parse_ledger`, `campaign_counts` and `g5_assemble` ignore. `final_outcome` and `outcome` carry the new divergent reason, so they agree with each other.
>
> **What it does change.** A parent killed at the memory limit, a repeated fault or an after-`AR_DEVICE` pattern now gives `discarded_divergent` instead of `infrastructure_failure_final`. That step is then:
> - audited (group A) and tabled (`TABLED`, with neg_inf / leave_out);
> - registered as evaluated, so a later re-proposal becomes `not_run_duplicate`;
> - counted in `divergent_in_campaign` and the single-seed-divergent share, and removed from `infrastructure_final` (STOP 7's share);
> - shown as divergent in the manager's history in history mode.
>
> Timeout counting is unaffected. These are pre-registration quantities.
>
> **Needed:** report item 13 discloses the change, but §7 does not list a note line for the author's signature. Add one (non-blocking for the harness).
>
> ## Other round-1 items
> - **m1:** report item 14 is corrected; the `read_rows` fix is listed as §7 item 1.
> - **m4:** catch-alls in `run_audit`, `run_campaign` and `main`. `stop_kind` is in both records and in the refusal record; ledger-derived kinds were tested.
> - **m5:** smoke items 10–15 cover the gaps I listed (systemd OOM handling, orphans, the CPU-only node, truncation on the real server, disk, the real job path).
> - **m6:** the paper's `campaign_file`, `campaign_diagnostics`, `tar_members` and `audit_chunk_file` now run on real engine output. The report wording is fixed.
> - **§7:** lists the paper-side follow-ups: `read_rows`, the deadline-to-STOP 9 mapping, the cgroup kill on `TimeoutExpired`, `OOMPolicy=continue`, and the `truncation_invalid_in` wording.
>
> Scratch is in /tmp/hg5rev/_to_delete/ (round-2 files under r2/).
>
> VERDICT: ACCEPTED

## Round 3 (`40396dc6 / 134b0399`)

> Reviewed base: 40396dc684980308f321a94280311dfc166a75a2 (G5-7, one commit on 94a41a7f) and 134b03994250367b2ee5ca2cd4940c929e1cb9c2 (G5-AR, the cherry-pick on 0aa27e31)
>
> ## Summary
> The confirmation round is clean, and the author's summary checks out on every point. The three non-blocking points from round 2 are all handled. Nothing new is needed on the harness side.
>
> ## Byte identity and pins
> - **The two families match.** `git diff 94a41a7f 40396dc6` and `git diff 0aa27e31 134b0399` are byte-identical.
> - **The pin files are untouched.** The pin delta `c025f89a..2d09f9a9` is identical to `40396dc6..134b0399`.
> - **File hashes match the author's figures.** `audit_g5.py` is 1089dbac… and the test file 469fb3ca… in both worktrees. `campaign_v2_diff.py` is unchanged (904e498b…).
> - **`g5_pin_check`** exits 0 at the new tips.
> - **`check_engines`** passes at both tips from all three checkouts. The negative control at `c025f89a` STOPs.
> - **`measurement_path_check`** passes for both families: 308 paths unchanged, and only the report, `audit_g5.py` and the test file are added. `engine_files_in_closure` is `[]` and the import closure is unchanged (15 modules).
>
> ## Tests
> One file at a time, identical in both worktrees; all three checkouts stayed clean.
>
> | File | Passed |
> |---|---|
> | `test_trackg_audit_g5.py` | 63 |
> | `test_trackg_campaign_v2_diff.py` | 16 |
> | `test_trackg_pilot_g4.py` | 31 |
> | `test_trackg_noise_floor.py` | 33 |
> | `test_trackg_isolation.py` | 28 |
>
> ## The new rule for log patterns after `AR_DEVICE`
> **What it does.** An infrastructure log pattern that appears only after the first `AR_DEVICE` line now has to happen twice before it counts against the code state:
> - The first time, it stays infrastructure and the run is retried. The run record is marked `edit_phase_pattern: true`.
> - If it happens again on a later attempt of the same code state and seed, the run is divergent, with reason `infrastructure_pattern_after_ar_device_repeated`.
> - If any default pattern appears before `AR_DEVICE`, the whole run stays infrastructure (this was round 2's nit).
>
> **Where the earlier attempts come from.**
> - In audits, the earlier attempts of the same planned (unit, seed) are checked; each attempt record carries its attribution.
> - In campaigns, `CampaignGuard.seen` keeps the flags per (code state, training seed).
>
> **It cannot loop.** If an edit causes the pattern every time, the second attempt is already divergent. Audits allow 3 attempts and campaigns 1 run plus 2 reruns, so that run never becomes an infrastructure row (no STOP 7) and never `infrastructure_failure_final`. A one-off host fault gets an ok rerun, so no false label.
>
> **My own probe of `attribute()`:**
> - pattern only after `AR_DEVICE`: infrastructure the first time, divergent `…_repeated` the second;
> - another default pattern before `AR_DEVICE`, the same pattern on both sides of it, or no `AR_DEVICE` line: always infrastructure;
> - an earlier crash signal does not count as an earlier pattern match.
>
> The new tests cover these cases in audits (the recurring edit pattern, a one-off fault whose rerun is ok, and a pattern before `AR_DEVICE` giving one infrastructure row after 3 attempts) and in one campaign.
>
> One observation, not a defect: two different default patterns in a row, both after `AR_DEVICE`, also count as a recurrence. For example, a full disk on the first attempt and an I/O error on the second. Both happen while the edited code runs, so this is acceptable.
>
> ## Report
> - **Note line for decision 4.** §7 item 6 proposes a `campaign_attribution` line for the author to sign. It is accurate: the three re-attributed cases become `discarded_divergent`, which is audited, tabled and counted in `divergent_in_campaign`.
> - **Persistent host fault.** It is disclosed in item 13 and in the §7 residuals.
> - **§9.** It maps the round-2 points to the fixes and tests correctly.
>
> Still open, and all on the paper side (report §7 items 1–6): the `read_rows` isolation checks, mapping a `deadline` stop to node-class STOP 9, killing leftover run cgroups after the engine is killed, `OOMPolicy=continue`, the `truncation_invalid_in` wording, and the signed note line. TASL-2 smoke items 1–15 must run before any EVIDENCE job.
>
> Scratch is in /tmp/hg5rev/_to_delete/r3/. The earlier round-1 and round-2 scratch folders were already gone from /tmp/hg5rev/_to_delete/ when I looked; I did not remove them.
>
> VERDICT: ACCEPTED

---

Code-Commit: 40396dc684980308f321a94280311dfc166a75a2
Verdict: ACCEPTED

Accepted at `94a41a7f` / `0aa27e31`, confirmed at `40396dc6` / `134b0399`.
