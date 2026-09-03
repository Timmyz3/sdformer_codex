# M2038 independent M2037 successor-result review

## Verdict

**PASS, 100/100; P0/P1/P2 = 0/0/0.**  M2037 is the sole fresh
successor attempt and canonical result.  There is no M2037 retry, failure
quarantine, or residual private stage.  The attempt marker, canonical result,
and all four upstream M2032/M2034/M2035/M2036 review trees verify against their
double seals.

This review admits functional VCS evidence for event-count/model-to-RTL
calibration on one real ep34 64-row **mask** tile.  The lane payload remains a
deterministic synthetic signed12 function of source and lane, and prior psum is
zero.  It is not real-weight or real-prior-psum numeric calibration.

## Compile, simulation, and counters

The unique compile and simulation return codes are both zero.  The exact full
terminal line occurs once and reports:

| Field | Value |
|---|---:|
| rows / active | 64 / 64 |
| input / residual nonzeros | 565 / 192 |
| exact-parent rows | 4 |
| issue accepts / parent edges | 196 / 58 |
| dead-write elisions | 31 |
| macro reads / writes | 54 / 33 |
| forwards / deadline holds / stalls | 4 / 6 / 14 |
| psum commits / row completions / numeric commits | 64 / 64 / 64 |

No compile/simulation error, fatal, assertion failure, watchdog, counter
mismatch, numeric mismatch, or protocol-error token occurs.  The receipt's
complete input/tool/upstream/log identity map matches independently recomputed
SHA-256 values.  The execution ledger remains one VCS compile, one `simv`, no
automatic retry, and a foundry `UNIT_DELAY` functional macro model.

## Exact publication and symlink repair

The canonical tree has exactly 96 manifest members, nine expected directories,
no unlisted regular files, no extra/unsupported filesystem objects, and zero
symlinks.  Both manifest and outer seal verify, and the manifest member set is
identical to the actual regular-file set.

`generated_symlink_removal.json` records exactly one removed VCS archive link:

`csrc/_2545240_archive_1.so -> .//../simv.daidir//_2545240_archive_1.so`

Its resolved target remains the regular in-tree file
`simv.daidir/_2545240_archive_1.so`, size 573,992 bytes, SHA-256
`83632f8b4f001e977ce3ed4b263a672e7834caa02e9910ca48fb0324da64a144`.
The recorded target, size, digest, raw spelling, and remaining-symlink count all
match the published tree.  Thus M2037 implements only the packaging repair
authorized by M2035/M2036.

## Old M2033 and claim boundary

The old M2033 canonical result remains absent.  Its original attempt marker and
single failure quarantine remain present, and the latter still contains
`FAILED_DO_NOT_CITE`.  No old output was salvaged into M2037; its diagnostic
PASS remains non-citable.

M2037 therefore closes only the narrow C1 calibration statement: for this one
64-row real-mask tile, the RTL reproduces the frozen model's service-event
counters and synthetic signed arithmetic.  It does **not** promote the M1590
`1.694510x` CPU cycle-model ratio to RTL speedup.  Same-area performance,
timing, power, energy, real-weight numeric calibration, full-network/system
speedup, and headline claims remain false.

No EDA, simulation, GPU work, or license query was launched by this reviewer;
the result, runner, RTL, TB, fixture, old M2033 artifacts, and docs/359 were not
modified.
