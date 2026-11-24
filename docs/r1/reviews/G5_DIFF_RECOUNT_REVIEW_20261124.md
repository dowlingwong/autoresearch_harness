Reviewed base: G5-7 `52e3c0b0` and G5-AR `00829da1` (the reviewed G.5 engines, READY 95); accepted at `e8a0dd65` / `04914a54`, not accepted at `db9c3d95` / `fd87eeea`, accepted at `a7646ec1` / `57434431`, confirmed at `c1063f6b50645587fc40b299411faf7c746fc8a8` / `e8d676a80f29160e014e7f538d34b66160d9e78b`

# Independent review record — R1-G5-DIFF-RECOUNT: manager diffs applied after recomputing hunk-header line counts from the hunk body (`git apply --recount` semantics), at both G.5 harness commits

**Unit.** In the harness repository, byte-identical G.5 changes on `mac-trackg-g5-7-20261124` (on `52e3c0b0`) and `mac-trackg-g5-ar-20261124` (on `00829da1`): `src/autoresearch/trackg/campaign_v2_diff.py` (the pre-pass `recount_hunk_headers`: header first — a hunk whose header counts fit its body is left as written, so every diff the strict parser accepted is byte-identical; otherwise the body decides (it ends at the next `@@`, a git file line or the end; a `---`/`+++` pair followed by `@@` ends it only in patches without `diff --git`); blank and one-space lines directly before the next file header or the end of the diff are separators (differs from git, disclosed); guards `no_old_lines` and `blank_only_anchor` and a whole-diff rule refuse hunks placed only by the header start and blank lines; only the two counts of `@@` lines of a copy change, and the unchanged `diff_support.apply_diff` applies the copy; context, offset and ambiguity matching are `diff_support`'s; malformed diffs stay refused); `src/autoresearch/trackg/audit_g5.py` (re-materialisation through the same pre-pass; `chunk_record.json` gains `diff_rule` and `diffs_recounted`, distinct diff texts); ledger lines gain a `diff_recount` block, `campaign_record.json` gains `diff_rule`; the ledger's `proposal.diff` stays the manager's text (O1 of D7-DIFF B); the run record's `diff_sha256` is the applied copy's hash where headers were rewritten (`diff_recount.raw_diff_sha256` holds the raw text's); new `scripts/trackg_g4_reclassify.py` (offline: verifies a sealed G.4 attempt against the committed captures, reads only `out/pilot.json` and the instance ledgers, rebuilds incumbents from the pinned node kit plus kept diffs checked against the ledger's code-state hashes, replays only parse/apply under the old and new rules; refuses a `--harness-root` whose HEAD differs from its own checkout); tests `test_trackg_diff_recount.py` (66) and `test_trackg_g4_reclassify.py` (9); report `docs/r1/reports/trackg_g5_20261124/G5_DIFF_RECOUNT_20261124.md`. `diff_support.py`, `runner.py`, `noise_floor.py`, `pilot_g4.py`, `manager_v2_diff.py` and `declared_edits.py` are unchanged; the paper's `g5_pin_check`, `g5_job.check_engines` and `g5_tasl.measurement_path_check` pass (308 measurement-path files unchanged; `noise_floor`'s 15-file import closure identical; no engine file in it). Authority: PLAN row R1-G5-DIFF-RECOUNT; the user's message of 24 Nov 2026 (18:10), item 1.

**Process.** One independent reviewer, read-only, own detached harness and paper worktrees, comparisons against `git apply --recount` in scratch repositories, fuzzing (300,000 random token diffs for header-first identity; about 30,000 real diffs with perturbed headers, blank separators and trailing blanks for wrong placements). Round 1 accepted with six minors; the coordinating session's m2 instruction (count trailing blanks as context, as git does) produced wrong-place applications and the first confirmation was not accepted; the author restored the separator rule and added fail-closed guards (0 wrong placements in 23,118 author fuzz diffs against 172 for round 1, liveness 66 % against 68 %), accepted with two wording minors, applied by the coordinating session after the author's session was lost in a server restart and confirmed.

| Round | Reviewed | Verdict |
|---|---|---|
| 1 | `e8a0dd65 / 04914a54` | ACCEPTED (minors m1–m6) |
| 2 | `db9c3d95 / fd87eeea` | NOT ACCEPTED (M1: trailing blank lines as context place hunks wrongly) |
| 3 | `a7646ec1 / 57434431` | ACCEPTED (minors m1, m2: wording) |
| 4 | `c1063f6b / e8d676a8` | ACCEPTED (confirmation) |

**Open items.** Disclosed residuals: (1) in a patch without `diff --git`, a removed `-- a/P` and an added `++ b/P` followed by `@@` send the next hunk to file P when its context matches (§3.5, tested); (2) a dropped trailing blank that was real context, next to a non-blank anchor, with a wrong header start line and a line identical to the anchor at that start, can place a hunk elsewhere (§3.3; 0 in 23,118 author fuzz diffs, 1 in 5,000 adversarial reviewer diffs; also under round 1). The separator rule differs from git `--recount`. The run record's `diff_sha256` is the applied copy's hash where headers were rewritten (no paper code compares it with `proposal.diff`). The pre-existing `test_trackg_t0_identity` failure is unchanged. The re-classification of the 122 G.4 `malformed_hunk` replies runs on TASL-2 against the sealed `out.tar` files (report §8; this acceptance covers the script, not its result; its `SHA=` names code commit `f31050f5`, behaviour-identical to the final tips, or use the final G5-7 tip). Paper side: the G.5 spec's `harness_families.*.g5_commit` take these tips (one more `key-lines` render of the G.5 draft); the G.4b re-pilot needs the paper-side pin rule in `g4_tasl.py`; both branches must be pushed before TASL-2 can fetch them.

## Round 1 (`e8a0dd65 / 04914a54`)

> Reviewed base: e8a0dd657 (G5-7, on 52e3c0b0) / 04914a54f (G5-AR, on 00829da1)
>
> ## Summary
> No blocking problems. The recount never placed an edit in the wrong spot in any test, apart from one contrived case the author disclosed (m1). P5 holds, the records follow O1, the tests match the report. Six minor findings.
>
> ## What I checked
> - **Same code in both families.** `git diff 52e3c0b0 e8a0dd65` and `git diff 00829da1 04914a54` byte-identical; so are `9eb0eb40` and `197d3ecb`.
> - **P5** (paper worktree at `3024db3f`): `g5_pin_check` exit 0; `measurement_path_check` exit 0 for both (308 paths, no engine file in the closure); `noise_floor`'s closure the same 15 files at both pins, both old tips and both new tips; `check_engines` passes; only six files changed; `diff_support.py`, `runner.py`, `noise_floor.py`, `pilot_g4.py` untouched.
> - **Private names** `_HUNK_RE`, `_ALLOWED_GIT_HEADERS`, `_REFUSED_GIT_HEADERS` exist and are used as the strict parser uses them.
> - **Compared with git.** ≈ 35 hand-built cases (headers too large, too small, omitted counts; several hunks and files with and without `diff --git`; a second file missing its `diff --git` line; trailing blanks, empty or single-space lines mid-hunk; `\ No newline`, CRLF; pure insertions/deletions, `-0,0`/`+0,0`; `-- `/`++ ` with and without `diff --git`) under old rule, new rule and `git apply --recount`. Whenever both the new rule and git applied, results were identical; other differences have known causes (git quirks: end-of-file anchoring of no-context hunks, swallowed `---`/`+++` pairs without `diff --git`; the deliberate `no_old_lines` refusal; the trailing-blank rule, m2).
> - **Fuzzing.** ≈ 4,800 real diffs with perturbed headers: none applied with a different result. 300,000 random token diffs: all 1,659 the strict parser accepts came back byte-identical.
> - **Context and refusals.** Only the two `@@` counts change; matching, offset search, ambiguity, scope and path checks are `diff_support`'s; non-count refusals keep their reason, except m4.
> - **O1 and the records.** `proposal.diff` stays the manager's text; the audit rebuilds through `recount_hunk_headers` (`audit_g5.py` `prepare_states`); no paper code (main, READY 103) compares the run record's `diff_sha256`/`diff_text` with `proposal.diff`; READY 103's `g5_assemble` hashes the raw text and its test builds run records with recounted `diff_text`; `g5_job` reads the campaign and chunk records with `.get`.
> - **G.4 behaviour.** G.4b recounts through `run_v2_diff_campaign`, as the row asks; nothing else changes. The paper-side blockers at `g4_tasl.validate_spec` line 152 and `launch` line 863 are real.
> - **Re-classification script.** Reads only `out/pilot.json` and the instance ledgers; trains nothing; calls no model; sound sha checks (out.tar vs both committed sums, SHA256SUMS and receipt byte-identical to the captures, job_record vs receipt); incumbents rebuilt and checked against the ledger's code-state hashes (normalised and raw); expected values agree with the G.4 report (122/33; 155 replays; 25 accepted = 22 ran + 3 duplicates).
> - **Tests** (both worktrees): diff_recount 54; g4_reclassify 5; diff_support 27; campaign_v2_diff 16; pilot_g4 31; noise_floor 33; isolation 28; audit_g5 63 (paper root set); runner 12; t0_identity 1 failed, 2 passed (same at `52e3c0b0`).
>
> ## Findings
> - **m1** `campaign_v2_diff.py:266-270` (docstring `:205-209`): in a patch without `diff --git`, a body ending with removed `-- a/model/b.py` and added `++ b/model/b.py` followed by `@@ -3,3 +3,3 @@` sends the next hunk to `model/b.py` (offset search at line 10); the old rule refused. Negligible as disclosed; the docstring overclaims ("refused (`context_mismatch`), never applied elsewhere" holds only if the first file lacks the line `-- a/<path>` there).
> - **m2** `campaign_v2_diff.py:276-278`: with a wrong header, trailing blank lines are separators, so a real trailing empty context line is dropped (applied where git `--recount` refuses); the same diff is refused when the header is too large by exactly the trailing blanks. Loosens matching by at most the trailing empty lines, never moves a hunk. The report should say it accepts some hunks git refuses.
> - **m3** `audit_g5.py:728`: `diffs_recounted` counts occurrences across units' chains, not distinct diffs. Rename or deduplicate.
> - **m4** recount docstring point 3, report §3.4: in multi-hunk diffs the refusal reason can change (hunk 1 too small, hunk 2 bad tag: old `malformed_diff`, new `malformed_hunk`); still refused; the script's `before` column comes from the ledger.
> - **m5** report line 218: "could not be kept" raw is overstated (`G5CampaignRunner` in `audit_g5.py` could override `_write_record`, at the cost of an inconsistent record and not covering G.4's default runner); the chosen design is better; fix the wording.
> - **m6** report §8 (≈ 343-357): `H` is written out; "final G5-7 tip" is not pinned (branches unpushed, no fetch/checkout step, `rev-parse HEAD` printed but not compared); the script does not check `--harness-root` is its own checkout; add `test "$(git -C "$H" rev-parse HEAD)" = <sha>`. The 122-reply re-classification has not been run (no committed capture holds a diff text); this acceptance covers the script, not its result. The `new_rule` counts are per step against G.4's actual incumbents, not a G.4b trajectory; label them.
>
> Note: header first goes beyond the row's literal "recompute from the body" only where it keeps diffs the strict parser already accepted; disclosed and acceptable.
>
> ## Housekeeping
> Worktrees removed and pruned; scratch in `/tmp/hrcrev/_to_delete/`. Nothing committed or pushed.
>
> VERDICT: ACCEPTED

## Round 2 (`db9c3d95 / fd87eeea`)

> Reviewed base: db9c3d957 (G5-7; code a983b424) / fd87eeea8 (G5-AR)
>
> ## Summary
> The m2 code change introduces a blocking regression: some diffs are now applied in the wrong place, including diffs the round-1 rule applied correctly. The two families are byte-identical, the P5 checks pass and every test passes. m1, m3, m4, m5 and m6 are fixed.
>
> ## Checks that pass
> Byte identity (`git diff 52e3c0b0 db9c3d95` = `git diff 00829da1 fd87eeea`). Paper checks from a fresh worktree at `3024db3f`: `g5_pin_check` exit 0 (same `g5_changes`); `measurement_path_check` passes for both (308 paths, no engine file in the closure); `check_engines` passes; `noise_floor`'s closure the same 15 files; `diff_support.py`, `runner.py`, `noise_floor.py`, `pilot_g4.py`, `manager_v2_diff.py` unchanged against `c025f89a`. Tests (both trees): diff_recount 59; diff_support 27; campaign_v2_diff 16; audit_g5 63; g4_reclassify 9; pilot_g4 31. In the 300,000-diff fuzz all 1,659 strict-accepted diffs still come back unchanged. m1: disclosure and test in place; agree a text-only refusal would reject every wrong-count multi-file patch in the prompt's format. m3: distinct diff texts by sha256. m4, m5 wording fixed. m6: the script refuses a mismatched `--harness-root`, records both commits and cleanliness, adds `basis`; `SHA=a983b424` sound.
>
> ## M1 (major) — trailing blank lines as context put hunks in the wrong place
> `campaign_v2_diff.py:272` (`stop_at = j`). With a wrong header, a blank separator (after a hunk before the next file header, or an extra blank at the end) becomes context; when it is the hunk's only context or the discriminator between candidate spots, `diff_support`'s exact-position-or-unique-offset search anchors the hunk on another blank line. Round 1 (separators) did not have this failure; git `--recount` usually places it the same way as round 2, but a wrong-place application is the main risk.
>
> Reproduced on a 10-line file with one blank line at line 2:
> - **C** (two-file `diff --git` patch; `@@ -6,0 +7,2 @@` inserting `+    h = 7`, then a blank, then `diff --git` for `model/b.py`): round 1 inserts after line 6 (correct); round 2 inserts at line 2 (a different edit; git does the same).
> - **A, A2** (`@@ -10,0 +11,3 @@` `+x = 1` followed by a blank and the next file, or an extra newline at the end): round 1 appends at the end (correct); round 2 at line 2; for A git refuses.
> - **B** (`@@ -10,1 +10,3 @@` `+x = 1` plus a trailing blank): round 1 refuses (`no_old_lines`); round 2 applies at line 2, the trailing blank defeating the `no_old_lines` guard.
>
> Fuzz (8,000 real `difflib` diffs with wrong headers, blank separators and trailing blanks): round 1 0 wrong applications; round 2 ≈ 2–5 per 1,000 (≈ 1 per 1,000 that round 1 applied correctly). Liveness drops ≈ 19–22 % of diffs round 1 applied correctly now refused with `context_mismatch`. A narrower patch (refusing hunks whose only context is blank) left wrong applications (a trailing blank can also pick the wrong spot among several matches for a real removed line).
>
> Required fix: return to round 1's handling (blank lines directly before the next file header or the end of the diff are separators in a recounted hunk), keep the disclosure that this differs from git (round-1 m2 asked for disclosure, not a code change), and add tests for A, A2, B and C asserting correct placement or refusal.
>
> ## Housekeeping
> Worktrees removed and pruned; scratch in `/tmp/hrcrev/_to_delete/` (round-1 rule `old/old_cv2d.py`; `adv/cmp.py`, `adv/m1case.py`). Nothing committed or pushed.
>
> VERDICT: NOT ACCEPTED

## Round 3 (`a7646ec1 / 57434431`)

> Reviewed base: a7646ec13 (G5-7; code f31050f5) / 574344312 (G5-AR)
>
> ## Summary
> Accepted. The restored separator rule and the new guards removed every wrong placement my round-1 and round-2 fuzz had found. The two families are byte-identical, the paper checks pass, and all six test files pass in both trees. Two minor findings, both about the report's wording.
>
> ## What I checked
> - **Byte identity.** `git diff 52e3c0b0 a7646ec1` = `git diff 00829da1 57434431`.
> - **Paper checks** (fresh worktree at `3024db3f`): `g5_pin_check` exit 0 (same `g5_changes`); `measurement_path_check` exit 0 for both (308 paths, no engine file in the closure); `check_engines` passes; `noise_floor`'s closure the same 15 files; `diff_support.py`, `runner.py`, `noise_floor.py`, `pilot_g4.py`, `manager_v2_diff.py` unchanged against `c025f89a`.
> - **Tests** (both trees): diff_recount 66; diff_support 27; campaign_v2_diff 16; audit_g5 63; g4_reclassify 9; pilot_g4 31.
> - **Header-first identity.** 300,000-diff fuzz: all 1,659 strict-accepted diffs byte-identical.
> - **Repro cases and git comparisons.** A, A2, C stop as `blank_only_anchor`, B as `no_old_lines`; none moved. The 37 earlier adversarial cases give the same old/new/git results as in round 1, including the disclosed `-- `/`++ ` residual.
> - **Fuzz** ("wrong" = applied with a result different from the reference): `adv/cmp.py` 10,000 diffs — round 1 8,602 correct / 1 wrong, new rule 8,142 / 0; blank-heavy variant 6,000 — round 1 4,256 / 24, new 4,397 / 0; blank-heavy with start lines shifted 1–3 scored against the correctly counted diff with the same starts, 5,000 — round 1 23 wrong, new 2 (one left as written with a fitting wrong header, read exactly as G.4; one the m1 residual).
>
> ## Findings
> - **m1** `campaign_v2_diff.py:316`: when a trailing blank that was real context is trimmed as a separator from a hunk that still has a non-blank old line, the hunk can be placed elsewhere than the correctly counted diff (needs a wrong header start, a line identical to the anchor at that start, and a real trailing blank context line). Constructed case: file `A = 1, z = 2, q = 3, A = 1, <blank>, w = 4`; hunk `@@ -1,5 +1,5 @@`, ` A = 1`, `+new = 0`, empty line: inserted after line 1; the correctly counted `@@ -1,2 +1,3 @@` inserts after line 4. Existed in round 1; rare. Fix: name it as a residual in §3.3 (line 107), the git table (line 186) and §13/line 538.
> - **m2** `campaign_v2_diff.py:155` (`_blank_placed`), line 294 (`weak`): the whole-diff rule does not flag a header-fitting hunk whose only old lines are removed blank lines, or an anchored header-fitting hunk whose fitting header counts a trailing separator as context; both behave as G.4's strict parser on correct headers, so the recount places nothing new. Narrow the wording ("keeps no blank-placed hunk") to "blank-context-only, or followed by a separator".
>
> ## Guard logic
> No gap where the recount itself places a hunk wrongly through the guards; the whole-diff rule fails closed; multi-file patches only lose liveness when one hunk is weak.
>
> ## Disclosure
> Liveness (66 % vs 68 %), guards, the `-- `/`++ ` residual, reason codes, the §7 test table and the departure note are accurate; only the absolute "never moved" / "0 different placements" claims need m1's qualification.
>
> ## Housekeeping
> Worktrees removed and pruned; scratch in `/tmp/hrcrev/_to_delete/` (`adv/cmp.py`, `cmp_b.py`, `cmp_c.py`, `adv/gap3.py`). Nothing committed or pushed.
>
> VERDICT: ACCEPTED

## Round 4 (`c1063f6b / e8d676a8`)

> Reviewed base: c1063f6b5 (G5-7) / e8d676a80 (G5-AR)
>
> ## Summary
> The wording commit is correct: it changes only the report and one docstring, both families are identical, the new text is accurate, and the paper checks pass on the new tips.
>
> - **The two diffs match.** `git diff a7646ec1 c1063f6b` = `git diff 57434431 e8d676a8`; from the reviewed G.5 bases, `52e3c0b0..c1063f6b` = `00829da1..e8d676a8`; each new commit's parent is the previously accepted tip.
> - **No code change.** Only the report and the `recount_hunk_headers` docstring in `campaign_v2_diff.py`; the parsed code with docstrings blanked is identical at `a7646ec1` and `c1063f6b`. `test_trackg_diff_recount` 66/66 in both trees.
> - **The wording is accurate.** §3.3 names the residual with my constructed example and both counts (0 in 23,118; 1 in 5,000), and says it existed under round 1. The git table row and §11 no longer claim "never moved" or an absolute zero; the "never moved" left in §13 refers only to A, A2, B, C and is true. The narrowed whole-diff wording in §3 point 4 and the docstring matches `_blank_placed` and the `weak` check and names the two unflagged shapes. Cosmetic: "Otherwise the whole diff…" now follows the inserted sentences and reads a little awkwardly.
> - **Paper checks** (fresh worktree at `3024db3f`): `measurement_path_check` exit 0 for both (308 paths, closure 15 files, no engine file); `check_engines` passes; `g5_pin_check` exit 0; `noise_floor`'s closure unchanged.
> - **Housekeeping.** Worktrees removed and pruned. Nothing committed or pushed.
>
> VERDICT: ACCEPTED

---

Code-Commit: c1063f6b50645587fc40b299411faf7c746fc8a8
Verdict: ACCEPTED

Accepted at `a7646ec1` / `57434431` (after a not-accepted confirmation), confirmed at `c1063f6b` / `e8d676a8`.
