# M1895 independent source hammer: M1894 minimal TSBG-B4 VCS one-shot

Verdict: **FAIL CLOSED (68/100, P0/P1/P2 = 0/5/2).** M1894 must not query a license, create its attempt, run VCS/simv, publish a result, or support a paper claim. No license, attempt, VCS, simv, DC, or PT action was performed by this review.

The reduced Bash runner does remove the Python alias/monkey-patch surface found in M1888 from the checked source itself. It has no `eval`, `source`, indirect command variable, or dynamic function dispatch; its literal launch sequence is attempt creation, one `lmstat`, one VCS compile with `-assert svaext`, and one newly generated `simv`. The M1880 filelist/RTL/adapter/SVA/TB identities and docs/359 identity re-hash correctly, all M1894 namespaces are absent, and `bash -n` passes.

Those positives are not sufficient to authorize the only attempt. Five P1 gaps remain:

1. **The external-call closure still trusts inherited Bash/PATH command bindings.** `env`, `sha256sum`, `awk`, `grep`, `find`, `sort`, `xargs`, `mv`, and other commands are resolved before the script establishes a clean shell namespace. An exported Bash function or an earlier `PATH` executable can fake hashes/seals and can replace the three `env -i ...` launch sites. In particular, `env -i` sanitizes the child only after Bash has resolved the unqualified `env` command. This reintroduces the zero/extra/fabricated external-call class that M1888 rejected.
2. **The independent review is not bound to the exact runner it authorizes.** The script checks a caller-supplied runner SHA and a caller-supplied review SHA separately, then only greps the review for PASS/count text. It never requires an audited runner SHA inside the sealed review to equal the live runner SHA. A changed runner can therefore be self-pinned by the caller while reusing a PASS review for an older runner.
3. **Success and failure publication do not prove that the move happened.** GNU `mv -T -n` returns success when it declines to replace an existing destination. Lines 149--150 can therefore leave `WORK` in place, set `WORK_ACTIVE=0`, and print publication success; lines 63--64 have the analogous failure-quarantine ambiguity. There is no source-disappeared/destination-sealed postcondition.
4. **Failure sealing is best-effort instead of fail-closed.** The EXIT path ignores both `seal_dir` and quarantine-publication failures with `|| true`. A post-attempt license/compile/sim failure can end with an unsealed or unpublished work directory even though the source comment promises one sealed non-retry quarantine.
5. **The same-UID process-name gate misses the Linux truncation of `common_shell_exec`.** `/proc/PID/comm` is limited to 15 visible bytes, so the normal basename truncates to `common_shell_ex`; the list contains only `common_shell_exec` and `common_shell_exe`. This concrete Synopsys collision can pass the gate.

Two P2 one-shot robustness gaps also remain:

- `REPO_ROOT` is computed but never entered. The frozen filelist contains repo-relative paths, so invocation outside the repository root consumes the unique attempt and makes VCS resolve the sources against the caller's current directory.
- `mkdir LOCK ATTEMPT WORK` and `seal_dir ATTEMPT` occur before `WORK_ACTIVE=1`. A partial mkdir/seal failure may consume the durable attempt without producing the promised terminal failure quarantine. This is fail-safe against repeat EDA, but not the stated terminal-ledger contract.

Required successor repair: keep this review FAIL-closed; do not edit or launch M1894. Author an additive runner that is launched from a clean, fixed shell environment and uses absolute non-EDA helpers; bind the sealed review's audited runner SHA to the live runner SHA; fix the `common_shell_ex` collision name; `cd` to the pinned repository root before VCS; create the attempt/work state transactionally; and require verified source-disappeared/destination-present seals for both success and failure publication. Then obtain a new different-author source review before any license query or attempt.

