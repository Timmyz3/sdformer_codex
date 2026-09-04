# M1333 — M1332 C2 production-activity source blind hammer

## Verdict

`FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED`

M1332 is not safe to promote to a mapped VCS release.  Its honest source-only
claim boundary is preserved, and several identities are correct: both M903
seals verify, the admitted K8/K1×8 mapped netlist and SDC hashes match, all ten
frozen cycle coordinates are represented, and direct diagnostic-K1 plus the
named legacy-memory mutations are rejected.  The author checker and all six
author tests also pass.

The independent hammer nevertheless found ten false negatives.  Four are
direct production-evidence failures:

- case 4 accepts manufactured endpoint toggles even though its frozen rule is
  exactly zero endpoint activity;
- activity under a sibling assertion scope can satisfy every major-cone gate
  while `core.dut` itself is empty, so the SAIF check is not DUT-only;
- a filelist may point to a forged same-leaf file under any path containing
  `/k8/netlist/`; the checker then hashes the unrelated official BASE object;
- the promised ten-file result is not executable—there is no uniqueness or
  complete `(axis, case)` inventory gate.

The source gate is also token-based rather than semantic.  A caller may omit a
required entry from `source_files`; after that, commenting out a memory reset
or a runtime cover still passes.  UCLI DUT commands left only in comments also
satisfy the scope checker while live commands capture `core`.  The old-memory
ban covers one spelling, not an exact compilation-unit/module-provider
allowlist.

Finally, the reset-safe memory does not meet its strongest stated invariant.
The state update branch accepts `mem_req_accept===1` plus known payload without
also requiring known asserted valid and ready.  A forced invalid/X-valid
accept can therefore index and modify slot state while merely setting a sticky
fault.  Payload/stability SVA are not explicitly fatal and are not folded into
the sticky unknown counter or a sealed future result gate.

## Required successor

The repair should be additive: exact resolved filelist path+SHA binding, an
exact source/compilation-unit allowlist, active-syntax checks, fully guarded
request/response state updates, hierarchy-aware DUT-only SAIF parsing, an
exact ten-file Cartesian inventory, and fail-closed assertion/cover terminal
accounting.  That successor needs another receipt-blind mutation hammer before
any VCS launch contract is authored.

No VCS, DC, PT, PTPX, GPU, or remote job was launched.  No RTL or M1332 source
was modified.  `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
