# M2190 independent M2189 source hammer

## Verdict

**FAIL, 91/100, P0/P1/P2 = 0/1/0. M2191 is not authorized.**

M2189 does repair the frozen M2181 defect: an independently reconstructed
fourth connected `/usr/bin/sleep` child is accepted by M2180 and rejected by
M2189. The declared one native plus fourteen process mutations pass, the
root/actual identity-collapse control passes, and normalized M2189 Tcl is
byte-identical to frozen M2180 Tcl except for milestone identity strings.

The new exhaustive-process claim is still not executable as written. The
production runner starts the monitored root before the launch gate and that
root repeatedly creates `/usr/bin/sleep 0.01` children. The pinned regular
`lm_shell` is itself a POSIX shell script which invokes external `dirname`,
`uname`, `cat`, `grep`, `cut`, and `basename` helpers before it launches
`lm_shell_exec`. Each helper is a connected process identity outside the
contract's allowed `set(root, actual, Milkyway)`. The checker correctly rejects
an independently injected bootstrap sleep below root, which proves that the
real runner topology and the synthetic three-process control disagree.

The monitor samples `/proc` every 5 ms. Therefore it has two unsafe outcomes:
it can observe a legitimate runner/wrapper helper and consume the unique run
with a false failure, or miss a short-lived helper and overclaim that no other
connected descendant ever existed. This is P1 because the proposed receipt
cannot support the promised exhaustive census.

## Checks that passed

- M2171, M2181, and the M2189 author receipt are exhaustive and double-sealed.
- Contract sidecar/outer seal and all six source identities match.
- Exact regular, non-symlink executable identities match for `lm_shell`,
  `lm_shell_exec`, `Milkyway`, and `lmutil`.
- All 1,051 Milkyway reference manifest members verify and no symlink exists.
- `lib.setting.milkyway_exec` is set and read back before the sole
  `generate_frame_from_mw` operation.
- No design import, P&R, timing, area, or power command appears in the Tcl.
- The M2181 `/usr/bin/sleep` failure lineage was reproduced; M2189 rejects it.
- A valid root/actual collapsed identity with a distinct Milkyway child passes.
- The official suite passes one native control, one native mutation, one
  process control, and fourteen process mutations.
- M2182/M2191 filesystem and running-tool censuses are empty. This review ran
  zero LM, EDA, license, GPU, or P&R action and did not modify `docs/359`.

## Required repair

Create a new source identity and do not execute M2191. Remove the child-sleep
launch gate or explicitly separate and verify the bootstrap/wrapper-helper
phase from the actual-LM subtree. Use event-complete child accounting if the
claim remains exhaustive; otherwise narrow the claim to sampled observations.
The actual-LM phase should remain fail-closed to exactly one pinned
`lm_shell_exec` and one distinct pinned `Milkyway`. Add a control representing
the real regular-wrapper topology and repeat an independent source hammer.

No M2191 LM run, license query, retry, design import, or P&R action is
authorized by this review.
