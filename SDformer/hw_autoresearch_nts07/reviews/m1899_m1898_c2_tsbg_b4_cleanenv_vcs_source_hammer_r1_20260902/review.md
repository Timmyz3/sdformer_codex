# M1899 independent source hammer: M1898 clean-environment TSBG-B4 VCS one-shot

Verdict: **PASS (98/100, P0/P1/P2 = 0/0/0)** with the exact status `PASS_M1899_M1898_C2_TSBG_B4_CLEANENV_VCS_SOURCE_HAMMER__AUTHORIZE_ONE_ATTEMPT`.

No license query, attempt creation, VCS compile, simv execution, DC, PT, or other EDA action was performed. This is source authorization only; it is not a VCS result and it admits no paper metric.

The audited runner SHA-256 is `35b4d25c907aa425e5b15d68d91be5f3fa4388f6e20ea7f48c2232ef6c0e1da6`. The filelist, RTL, adapter, SVA, TB, predecessor review, and docs/359 identities all re-hash to the literals in `review.json`; the M1895 directory double seal verifies. All M1898 attempt/result/failure/work/lock namespaces were absent during review.

## M1895 regression

All five P1 and both P2 findings are closed:

1. The executable shebang uses `/usr/bin/env -S -i` with fixed `PATH`, `LANG`, and `LC_ALL`; helpers and EDA tools are absolute. A direct harmless zero-argument probe under a poisoned inherited PATH and exported Bash function reached the runner's normal argument error without executing the poison.
2. The runner accepts only two 64-hex identity arguments, verifies the live runner and sealed review independently, and requires the review to name the same runner SHA. A future release must pin this exact pair.
3. Success and failure publication both prove source disappearance, non-symlink destination presence, and both destination seals after `mv -T -n`.
4. Failure sealing/publication are no longer hidden by `|| true`; the trap clears itself before terminal processing.
5. The same-UID blocklist includes the actual 15-byte procfs truncation `common_shell_ex`.
6. The runner enters the fixed absolute repository root before consuming the attempt.
7. The work directory and failure handler become active before attempt creation, so a partial attempt setup is terminally quarantined and the durable attempt forbids retry.

The literal execution shape is one sealed attempt, then exactly one `lmutil lmstat`, one VCS compile with `-assert svaext`, and one work-local `simv`. There is no `eval`, sourced code, indirect executable dispatch, dynamic untrusted path, or automatic retry. The only recursive deletion targets are three fixed descendants of the PID-scoped absolute work directory.

## Launch boundary

The future release must execute the absolute runner **directly** under a clean environment and pass the exact runner SHA plus the live SHA-256 of this `review.json`. It must not use `/bin/bash runner`: explicit-interpreter invocation bypasses any script's kernel shebang and is outside this authorization. A different-author release audit is required before the sole license query or attempt.

This PASS authorizes only one directed behavioral RTL compile/simulation campaign. Even a raw simulation PASS must await a different-author result hammer. Area, energy, same-area comparison, component/system speedup, and paper admission remain false.
