# M2154 independent hammer of the M2153 ICC2 preflight source

## Verdict

**FAIL source gate; M2155 is not authorized.**  M2153 repairs the M2146 P0
startup-scope defect and every specifically listed P2 item.  It also materially
strengthens the root inventory, union-94 coverage, gate grammar, process census,
and author seal.  However, two independent anti-forgery mutations still survive
the raw parser: a NUL-prefixed invented file is accepted as a native NDM/design
database, and a disconnected two-process parent cycle can carry the alleged
`dgcom_exec` observation without descending from the monitored root.  These are
two P1 evidence-boundary defects.  The required P0/P1/P2 = 0/0/0 admission rule
therefore prohibits the one M2155 launch.

M2154 invoked no ICC2/EDA executable, no license client, and no GPU.  It did not
edit the M2153 source, paper, M2141/M2147, or protected `docs/359`.  The latter
still hashes to
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Repairs that independently reproduce

The mechanical audit completed 1,141 positive checks:

- all M2153 sources match their author-receipt identities, and both the M2146
  predecessor and M2153 author receipt are path-exhaustive, symlink-free,
  double-sealed directories;
- the shell has exactly one `lmutil lmstat` site and exactly one
  `icc2_shell -no_init -f <exact-Tcl>` site, under `env -i`, isolated
  `HOME`/`TMPDIR`/`XDG_CACHE_HOME`/cwd, with both the wrapper and resolved
  1.7-GB `dgcom_exec` SHA-pinned;
- the Tcl contains one non-overwriting frame conversion, one disposable
  design-library creation, exact `core`, M1--M9, VIA1--VIA8, TT/SS/FF and RC
  checks, and no RTL import, compile, placement, CTS, route, timing, area, or
  power command;
- the independently re-derived mapped-netlist union equals the frozen sorted
  94-master list, and all 1,051 Milkyway files rehash against the exhaustive
  manifest with no symlink;
- the monitor now includes installed ICC2 and conversion executable names and
  records PID/start-time identities, parents, exec transitions, and selected
  isolation variables;
- the root inventory handles regular files, directories, links, FIFOs,
  sockets, block/character devices, and unknown nodes; runtime probes confirmed
  all user-creatable classes;
- wrong master names, pure-text databases, a changed root inventory, and a
  missing actual-exec `-no_init` are rejected; and
- M2155 has no attempt or result directory, so the one-shot budget remains
  unconsumed.

These checks close the original startup-file, executable-pin, wildcard-core,
top-level-node, omitted-process-name, summary-count, and pycache findings.

## P1-1: a NUL byte is not a native NDM discriminator

`tree_stats()` labels any file whose first 64 KiB contains `\0` as binary.
The independent fixture supplied directories named `.ndm` and `.nlib`, each
with one invented `native.db` containing only a leading NUL and forged text;
it also supplied invented 100-byte reports.  The complete raw checker returned
PASS and reported one binary member for each library.

This does not meet the contract's native-database predicate.  Installed NDMs
from the same V-2023.12 library manager are regular `*.ndm` members with the
Synopsys Library Manager header (the inspected `gtech.nlib/reflib.ndm` starts
`b2 bd ea 03`), not arbitrary NUL-bearing files.  The parser already rejects
pure text and empty trees, but that only proves non-text, not native format.

Minimum repair: require the generated frame to be a regular `.ndm` with the
same-version Library Manager magic/header, require the design `.nlib` to
contain a regular native `reflib.ndm` (or the exact documented native member
layout), and reject the current directory-plus-`native.db` synthetic fixture.
Keep the object reports and exact runtime gates as additional, not substitute,
evidence.

## P1-2: local parent links do not prove root reachability

The checker verifies that each non-root identity has *some* observed parent.
It does not compute transitive reachability from `root_pid`.  The independent
mutation moved the exact `dgcom_exec` observation into identities 101 and 102,
made 101 name 102 as parent and 102 name 101 as parent, and left identity 100 as
the monitored root with only a wrapper observation.  All summary counters and
lists remained self-consistent; the raw checker returned PASS with three
identities.

The live monitor only samples descendants, so this mutation attacks evidence
integrity rather than predicting a normal live trace.  That is nevertheless the
same fail-closed boundary M2146 required the raw parser to enforce.

Minimum repair: construct the directed identity graph from exact
`(pid,starttime)` parent links, require every non-root identity and every actual
ICC2 observation to be in the transitive closure of the unique root, reject
cycles/disconnected components, and require a parent's start time not to exceed
its child's start time.

## Score and authorization

- Score: **91/100**
- P0/P1/P2: **0/2/0**
- M2155 authorized: **false**
- license queries / top-level ICC2 / P&R authorized: **0 / 0 / 0**
- library import proven: **false**
- RTL/P&R/timing/area/power or paper-PPA evidence: **false**

The repair is parser-local.  Do not alter or retry M2141/M2147/M2155.  Use a
new additive source identity, rerun an independent source hammer, and authorize
one fresh library-only attempt only if that successor scores at least 95 with
P0/P1/P2 = 0/0/0.
