# M873 — M872/M803 C2 R16 three-axis DC source hammer

**FAIL 98/100; P0/P1/P2 = 0/0/1.  Return to author; no release and no DC.**

The executable no-EDA closure is otherwise strong.  Independent replay passed
all 17 exact-file hashes, the 12-entry regular nonsymlink RTL filelist,
36-function define-before-use closure, and the exact `K1/K8/K1x8 =
ARCH_MODE 0/1/2` bindings.  The Tcl orders analyze, elaborate, precompile
`check_design`, `check_timing`, and the fatal `TIM-209=0`/`OPT-150=0` gate
before its single `compile_ultra`.  Seven outputs per axis—mapped Verilog,
mapped SDC, DDC, SVF, area, QoR, and setup timing—are required by the live
receipt and enclosing-manifest checks.

The full candidate-to-provenance-to-contract path executed successfully in a
clean HOME-absent no-EDA environment and exited before resource preflight,
license query, attempt publication, or tool launch.  The atomic artifact test
passed one positive and 25 deletion/zero/symlink/ancestor/path-escape/partial-
publication/post-receipt-mutation negatives.  Wrong runner SHA returned 3 and
created one valid double-sealed pre-attempt failure receipt.  Python 3.6 and
the host's independent modern Python 3.12 interpreter agree; Python 3.10 is
not installed on this host.  Canonical, attempt, work, and quarantine
populations remained zero.

## P2 finding

`P2_STALE_R15_CURRENT_SUCCESSOR_TERMINOLOGY`: the new M872/R16 source identity
still labels several *current successor* obligations as R15, even though R15
is the consumed predecessor that M800 forbids reusing.  In particular:

- candidate field `all_three_axes_must_rerun_under_one_r15_attempt` is false
  for the hard-coded M872/R16 attempt;
- candidate `required_next_gate` asks for a fresh R15 source hammer rather
  than this M872 hammer;
- five contract `forbidden` entries tell the reader to run/consume/release the
  R15 package or set its HOME environment, instead of naming M872/R16.

The runner's hard-coded M872 canonical/attempt identities and per-axis calls
prevent this wording defect from becoming an executable bypass, so this is P2
rather than P0/P1.  It still violates the explicit pass rule requiring
P0/P1/P2 = 0/0/0 and makes the sealed contract internally inconsistent.

The minimum successor is source-only: replace only those prospective/current
R15 labels with M872/R16 labels, retain legitimate historical `r15_*` M800
provenance unchanged, issue new runner/contract/candidate identities and
seals, and request a fresh source hammer.  Do not edit this sealed target in
place and do not author `launch_now=true` yet.

This review invoked no DC, VCS, simulator, license query, PT, PTPX, Formality,
remote job, or production runner path.  It did not modify M872 sources or
`docs/359`.
