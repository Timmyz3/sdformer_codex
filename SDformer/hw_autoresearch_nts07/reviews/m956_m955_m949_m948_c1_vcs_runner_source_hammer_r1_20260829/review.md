# M956 | M955 C1 VCS runner source hammer

Verdict: `GO`, review status `PASS_M956_M955_VCS_RUNNER_SOURCE_HAMMER`,
score 98/100, P0=0/P1=0/P2=1. M956 does not release or run VCS.

The sidecar-cwd defect is genuinely removed. Five extant double-sidecar pairs
(M955 runner, M948 DRAFT, M951 predecessor contract, the M951 failure receipt,
and M955 contract) validate when `sha256sum -c` is executed from `HW_ROOT`.
M955 has exactly five `verify_hw_root_sidecar_pair` call sites; all members are
explicit `contracts/...` or `results/...` root-relative paths. The fifth future
call is the required M957 release. M957 remains absent, so launch remains
forbidden. The inherited `cd contracts` plus `basename` verification pattern
occurs zero times.

The sealed M951 receipt confirms exit code 1 at pre-attempt sidecar resolution:
no attempt, result, work directory, VCS compile or simv run was created. Current
filesystem state independently still shows all M951 paths absent, and M951
rerun/reuse remains forbidden.

After substituting only M955/M951 identity strings, the 4,823-byte execution
tail from the unique-attempt guard through success sealing is byte exact. Thus
the one-attempt guard, same-UID EDA collision check, 64-GiB memory gate,
PIPESTATUS checks, timeout, failure quarantine, runtime tokens, receipt fields,
and false timing/cycle/speedup/PPA/energy/system claims are preserved.

Exact RTL, macro wrapper, inherited SVA, TB, checker, foundry model, VCS binary,
M948/M949 identities, consumed M943 evidence and docs359 all validate. The M948
static checker passes. At audit time same-UID EDA hits were zero and
MemAvailable was 421,507,656 KiB.

P2: live process and memory state remains transient and must be rechecked by the
released runner. No P0/P1 was found. A separately sealed M957 must bind this
exact review and M955 identities before one functional attempt may launch.
Nothing here admits functional success, timing, workload cycles, speedup, PPA,
power, energy, system, headline or paper claims.
