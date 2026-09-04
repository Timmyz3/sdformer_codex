# M547 / M533 r6 repair-author handoff

Status: **source-only complete; no launch authorized**.

- TB r4 is a strict mechanical repair: exactly four lines rename the illegal local SystemVerilog identifier `packed` to `packed_row`. Core r2, SVA r2, macro adapter, binding plan, test behavior, module name, and tokens are unchanged.
- Runner r6 uses a new result/attempt identity and never writes the old consumed r3 partial. Every post-`mkdir` failure is routed through the EXIT trap to `RUN_FAILED_OR_INCOMPLETE.json`, `FAILED_DO_NOT_CITE`, a recursive member manifest, and an outer manifest seal. A genuine functional success also receives a double-sealed receipt.
- The runner records phase, runner/child return codes, monitor status, resource/collision status and hashes, recursive pre-receipt inventory and hashes, and immutable source hashes.
- Source contract, static-review request, and `launch_now=false` admission candidate are double sealed. The new result path and all future PASS/release members are absent.
- Author execution count is zero for runner, VCS, simv, all other HDL/EDA tools, experiments, and remote jobs.

The only legal next step is the fresh independent source-static hammer defined by the request. A 100/100, P0/P1/P2 = 0 result still does **not** authorize VCS: candidate hammer, a separate final release, and a fresh final-release hammer must follow.
