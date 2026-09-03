# M1905 independent release audit: M1904/M1898 TSBG-B4 directed VCS

Verdict: **PASS (99/100, P0/P1/P2 = 0/0/0)** with exact status `PASS_M1905_M1904_C2_TSBG_B4_LAUNCH_RELEASE_AUDIT__AUTHORIZE_ONE_M1898_ATTEMPT`.

This was a static release audit only. I made no license query, attempt, VCS compile, simv run, DC/PT run, or result. The M1898 attempt, work, lock, result, and failure namespaces were absent at audit time, so no pre-M1905 launch was observed.

The M1904 release SHA-256 is `6a3301937ea1e9dc090173dffd41d29644250ecff327777bc3450a055dac5d7b`; its inner and outer seals verify. It pins the exact executable runner SHA `35b4d25c907aa425e5b15d68d91be5f3fa4388f6e20ea7f48c2232ef6c0e1da6` and the exact M1899 review SHA `0214a788756bd6d2c0fd5c1e8c900c58274e34e75ab3d0ab7e856e192ac374f6`. The M1899 manifest/outer seals, every source identity including the M803 adapter, and frozen docs/359 re-hash to their audited values.

## Authorized invocation

Only one invocation shape is authorized: execute `/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_m1898_m1880_c2_tsbg_b4_cleanenv_directed_vcs_one_shot.sh` **directly** and pass exactly the pinned runner SHA followed by the pinned M1899 review SHA. `/bin/bash runner ...` is forbidden because it bypasses the kernel interpretation of the clean `env -S -i` shebang.

The frozen runner consumes and double-seals the attempt before its first external license/EDA action. Its literal budget is one `lmutil lmstat`, one VCS compile with `-assert svaext`, and one work-local `simv`; there is no retry. The raw success receipt keeps same-area, component/system speedup, and paper admission false.

## Boundary

This PASS authorizes exactly one M1898 directed behavioral compile/simulation attempt. It does not establish a VCS PASS, timing, area, energy, or speedup. Success or failure must be sealed, and even a raw simulation PASS requires a different-author result hammer before it may be admitted.
