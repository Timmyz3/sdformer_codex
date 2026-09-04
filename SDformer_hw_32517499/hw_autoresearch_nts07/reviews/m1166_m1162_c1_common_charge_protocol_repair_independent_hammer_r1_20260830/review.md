# M1166 — M1162 common-charge protocol repair independent hammer

Verdict: **PASS the repaired wrapper source; authorize exactly one additive
VCS TB/SVA/filelist/launcher source package, but do not run the current TB and
do not run VCS or EDA yet.**

## What passed

- The nested author and contract seals resolve. Frozen M935 remains
  `e834b524...`, the parent macro wrapper remains `8fd008a3...`, the common
  charge map remains `16da0132...`, and `docs/359` remains `dedde7ce...`.
- The ledger is unchanged and exact-once: `18,432 B` internal parent SRAM plus
  `122,880 B` external psum, `49,152 B` external weight, and `24,448 B`
  external metadata reserve equals `214,912 B`, leaving `30,848 B` under
  `240 KiB`. Only the nine parent macros are physical in this component.
- Both request valids are functions of M935's held request and registered
  acceptance state; neither has `ready` in its combinational fan-in. All 16
  pairs of legal own-valid-dependent ready functions therefore evaluate
  without the M1116C combinational-loop defect.
- Weight and first-beat psum requests have separate accepted flags. After its
  own handshake, each valid is suppressed while its peer remains live. The
  exhaustive 32-state by 256-input check found no accepted-request reissue.
- A response cannot be accepted before its own request. First-beat response
  consumption is a joined atomic handshake; either skew order and arbitrary
  core backpressure are supported by relying on the frozen response-hold
  service rule. No 1,152-bit or 1,824-bit payload FIFO was added.
- Reset cancels the depth-one transaction. Same-cycle/unsolicited/early
  response, non-first psum response, request cancellation, and request-tuple
  mutation reach the sticky boundary fault. Eleven independent source
  mutations were rejected.
- The state accounting is truthful: `36` request-tuple bits plus active, two
  accept flags, and sticky fault equals `40` bits. With zero stalls and a
  one-cycle service response, completed issues have minimum `II=2`.

## No P0; one validation P1

No wrapper protocol P0 was found. The current M1162 TB is nevertheless a
source sketch, not the executable verification gate described by its own SVA
plan. It does not yet execute a non-first beat, an explicit II=2 recurrence,
reset in request-complete and response-skew states, unsolicited psum response,
duplicate-request mutation, or response-payload hold/mutation checking. The
SVA is still a Markdown plan rather than an executable assertion module.

This is a verification-source P1, not a reason to reject the repaired RTL.
Directly compiling the current TB would produce an incomplete receipt and must
not be used as the M1162 VCS admission.

## Exact next authorization

Exactly one additive source-only VCS package is authorized. It may add a new
TB, executable SVA, synthesis-free VCS filelist, and exact-SHA launcher. It
must bind the frozen M1162/M935/macro identities and execute all plan cases,
including both partial request orders, both response orders, valid-dependent
ready, long stalls/backpressure, non-first, II=2, three pending-reset states,
spurious/early/cancel/mutation attacks, response payload hold, and duplicate
request suppression.

That package must remain `vcs=false` until a fresh hammer checks its source and
explicitly authorizes one VCS run. DC/PT/Formality/PTPX and matched performance
replay remain unauthorized. No cycle, speedup, numeric external-memory area or
energy, full-storage, system, or paper-ready claim is created here. In
particular, M1114's `1.7591725402×` is not inherited.
