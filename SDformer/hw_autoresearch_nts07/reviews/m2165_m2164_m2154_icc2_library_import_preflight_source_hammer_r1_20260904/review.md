# M2165 independent hammer of M2164 ICC2 library-preflight source

## Verdict

**PASS source gate, 98/100, P0/P1/P2 = 0/0/0.**  The exact M2164 source at
commit `352ee1c1` closes every defect reported by M2146 and the two remaining
P1 defects reported by M2154.  M2165 authorizes exactly one M2166
library-import-only preflight: one license query, one top-level `icc2_shell`
session, zero P&R runs, and no automatic retry.  It does not authorize P&R.

The review invoked no ICC2/EDA executable, license client, or GPU process and
did not modify the source, paper, or protected `docs/359`.  The latter remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Independent evidence

The mechanical hammer passed 1,153 checks.  It rehashed the contract, runner,
parser, immutable Tcl, monitor, inventory, union-94 list, two predecessor
reviews, same-version NDM reference, all 1,051 members of the Milkyway
reference, and the protected document.  M2146, M2154, and the M2164 author
receipt remain exhaustive, symlink-free, double-sealed directories.  Neither
the M2166 result nor its attempt-consumed marker exists.

The runner has one exact `icc2_shell -no_init -f` site and one `lmutil lmstat`
site.  It executes from an isolated cwd through `env -i`, with isolated HOME,
TMPDIR, XDG cache, and library cache; pins both the wrapper and selected
`dgcom_exec`; creates and seals the attempt marker before either authorized
external operation; and quarantines any failure.  The immutable Tcl has one
non-overwriting frame conversion, one disposable design-library creation,
exact `core`, and no RTL import, synthesis, placement, CTS, route, timing,
area, or power command.

## Native database predicate

The installed V-2023.12-SP3 `gtech.nlib/reflib.ndm` rehashes to the pinned
identity and begins with the exact 68-byte native Library Manager header:
magic, `Library Manager`, release, platform, and build date.  M2164 requires
the generated frame to be a regular nonsymlink `.ndm`, and the design library
to be a nonsymlink `.nlib` directory containing regular nonsymlink
`reflib.ndm`; both native members must match that header.  Independent attacks
using an arbitrary NUL-prefixed file, a truncated header, a different release,
a directory named `.ndm`, and a wrong suffix were all rejected.

This is not treated as a standalone authenticity oracle.  It is conjunctive
with the immutable tool-generated path, runtime object queries, exact gate
tokens, frozen four-view master coverage, object reports, nonempty tree
statistics, and the mandatory M2167 result review.  The authorized result, if
successful, remains library compatibility only and is not P&R or paper PPA.

## Process provenance predicate

The checker reconstructs the complete observed `(pid,starttime)` graph, admits
one root identity, forbids an internal parent of that root, requires an exact
observed internal parent for every non-root, checks every internal parent's
start time, performs a complete three-colour cycle check, and requires every
identity to be transitively reachable from the root.  Summary lists and counts
must equal the flattened census; the selected `dgcom_exec` path, exact Tcl,
`-no_init`, and isolated environment are rechecked.

Independent mutations for a disconnected cycle carrying the exact
`dgcom_exec`, a reachable cycle hidden behind an extra edge, a parent starting
after its child, an orphan, a duplicate root PID identity, forged summary
counts, and the wrong actual-executable environment were all rejected.

## Authorization boundary

- M2166 library-only preflight: **authorized exactly once**.
- License queries: **1**.
- Top-level ICC2 sessions: **1**.
- P&R runs: **0**.
- Automatic retry: **false**.
- M2167 independent result hammer: **required**.
- New matched P&R source or execution: **not authorized by M2165**.

Any failed or incomplete M2166 attempt is terminal for this identity.  A PASS
from M2166 is raw evidence pending M2167, not a physical result.
