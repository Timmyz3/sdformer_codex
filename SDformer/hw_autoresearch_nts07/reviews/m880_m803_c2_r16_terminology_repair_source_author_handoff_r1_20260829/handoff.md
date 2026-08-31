# M880/M803 C2 R16 terminology-repair source handoff

M880 is the minimal additive repair requested by M873.  The five prospective
contract prohibitions now name the current M872/M803 R16 attempt or the M880
source package.  The candidate same-attempt key names M872/M803 R16, and its
required next gate names the fresh M880/M803 R16 hammer.  Historical `r15_*`
M800 and artifact-publication provenance remains byte-for-byte semantic
history and was not relabeled.

Because the runner binds the contract path and the contract binds the runner
SHA, M880 uses a fresh runner/contract/candidate identity.  The runner retains
the frozen M872 canonical and attempt, all M803 source and flow inputs, and the
same production EDA behavior.  Its strict JSON pre-attempt parser additionally
rejects `NaN` and infinities.

The full no-EDA closure passes under Python 3.6.8 and Python 3.10.18.  It covers
three duplicate-key and three nonfinite JSON negatives, 19 semantic mutations,
one aggregate stale-R15 reinjection, the exact full admission path, one
positive plus 25 negative artifact-publication cases, and the double-sealed
wrong-runner-SHA failure path.  Canonical, attempt, work, and quarantine
populations remain absent.

This handoff authorizes only a fresh independent M881 source hammer.  It does
not authorize DC, VCS, a license query, PT, PTPX, Formality, remote work, or a
launch release and makes no physical or performance claim.
