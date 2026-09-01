# M1796 C2 registered public-fault successor — author receipt

Status: `PASS_M1796_SOURCE_AUTHORING__READY_FOR_DIFFERENT_AUTHOR_M1797_HAMMER__NO_EDA`.

## Root-cause boundary

The frozen M803 K8 top publicly ORs four sources: `core_protocol_error`,
`adapter_protocol_error`, `consistency_fault_q`, and the current-cycle
`consistency_fault_now`.  The last source compares request and response accept
copies with unconditional four-state `!=`; it is not qualified by the owning
valid.  M1785 first sees the mapped public pin become X at 26000 ps while all
eight endpoint faults and all six selected registered internal fault taps are
zero.  This is strong source-plus-diagnostic consistency, but it is not called
a successor VCS confirmation.

The core and channel-split adapter public fault outputs also contain
current-cycle `illegal_*` predicates in addition to sticky state.  M1796 thus
does not merely remove one OR input.  It introduces a synthesizable registered
public-fault boundary: core/adapter events are observed under explicit owner
activity, request/response accept mismatch is sampled only under its valid, and
the public pin is driven only by resettable sticky state.

## Why true faults are not hidden

- Illegal header/raw/response predicates and their original ready/accept gates
  remain inside the frozen children; M1796 changes no child data or handshake
  logic.
- An asserted illegal header, illegal raw packet, or spurious response is
  rejected at its original boundary and sampled into sticky fault state at the
  next edge.
- Internal request/response accept disagreement is still sampled whenever the
  corresponding transaction is valid.  Only don't-care payload under
  `valid=0` is excluded.
- No force, initreg, ignore-X, assertion suppression, case-equality RTL
  coercion, or constant-zero fault implementation exists.

## Source verification prepared

The unit TB covers legal case0, X-valued invalid payload under `valid=0`, four
independent event inputs, half-cycle plus posedge/negedge binary checks, sticky
behavior, and reset recovery.  The cloned full-top regression retains the
five exact K8/K1x8 cycle pairs, numeric/tuple/weight equality, backpressure,
out-of-order responses, reset attacks, and now explicit accept-zero checks for
illegal header, illegal raw, and spurious response attacks.

The fail-closed checker and 19-mutation suite pass under CPython 3.6 and 3.12.
No EDA, attempt, result, commit, or push was performed.

## Mandatory downstream closure

After a different-author M1797 hammer, run both RTL VCS campaigns.  If they
pass, the source identity change invalidates the old physical comparison:
resynthesize at least K8 and equal-bandwidth K1x8 under identical constraints,
then rerun Formality, mapped VCS, SAIF, and PTPX.  M1661 area and prior energy
must not be reused as M1796 evidence.
