# M2146 independent hammer of the M2141 ICC2 library preflight source

## Verdict

**FAIL source gate; the exact M2141 source does not authorize M2147.**  The
intended library-only sequence is materially better than M2135 and most of its
frozen identities independently reproduce.  However, the only ICC2 command is
invoked without the documented `-no_init` option.  Consequently an unsealed
`.synopsys_icc2.setup` can execute before the reviewed Tcl and can import RTL or
invoke implementation commands outside the stated zero-P&R budget.  `env -i`
does not close that boundary because `HOME` is not redirected to the isolated
work directory and the executable has its own startup-file search behavior.
This is one P0 execution-scope defect.  M2147 must not be launched from this
source identity.

No ICC2 executable, license client, or GPU process was invoked by M2146.  The
protected `docs/359` SHA remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## What independently reproduces

The independent mechanical audit passed 1,137 positive identity and structural
checks before issuing the fail verdict:

- the M2141 contract, runner, Tcl, monitor, checker, 94-master list, and
  protected document match their recorded SHA256 identities;
- both M2029 mapped netlists rehash exactly, and a broader independent Verilog
  instance grammar re-derives the same sorted union of 94 standard-cell
  masters (91 ordinary, 81 TSBG, 94 union);
- the frozen Milkyway tree contains exactly 1,051 regular files, its manifest
  is path-exhaustive, every listed file rehashes, and no symlink exists;
- the TT/SS/FF DBs, NXTGRD, layer map, ICC2 wrapper, `lmutil`, and the locally
  selected real ICC2 executable (`dgcom_exec`) rehash exactly;
- the M2135 attempt and quarantine and the M2136 review still have valid,
  exhaustive double seals.  The raw M2135 log contains exactly one each of
  CMD-104, LIB-117, FILE-001, and LIB-027, terminates at the first
  `create_lib`, and never starts the TSBG axis;
- the M2141 Tcl uses the documented `set_app_options`/`get_app_option_value`
  pair, one non-overwriting `generate_frame_from_mw`, the converted frame as
  the only physical reference, exact TT/SS/FF `link_library` inputs, and no
  explicit RTL import, `compile_fusion`, placement, CTS, or route command;
- the source contains one `lmutil lmstat` site and one top-level ICC2 site, with
  no retry loop.

These checks validate the intended repair but do not neutralize commands that
ICC2 can run from an initialization file before the reviewed Tcl begins.

## P0: startup files remain outside the sealed command budget

The installed V-2023.12-SP3 `icc2_shell(1)` reference explicitly defines
`-no_init` as the option that prevents `.synopsys_icc2.setup` files from being
run.  M2141 executes:

```text
"${ICC2}" -f "${TCL}"
```

and not `"${ICC2}" -no_init -f "${TCL}"`.  The runner also leaves `HOME`
unset rather than assigning an isolated empty home.  No startup file happens
to be present in the inspected home/cwd today, but its absence is neither
hashed by the source review nor enforced immediately before execution.  Thus
the claim `rtl_import=false`, `pnr_runs=0` is not source-enforced.  This is a
scope defect, not a prediction that the current installation will necessarily
misbehave.

## Five P1 evidence-boundary defects

1. **The raw checker accepts forged semantic evidence.**  A synthetic fixture
   with arbitrary `M2141_GATE1_*` text, a wrong master substituted into the
   94-row coverage table, empty frame/design-library directories, fabricated
   but equal repository snapshots, and no real ICC2 output is accepted.  The
   checker only matches gate-number prefixes, sorted row counts, and path
   existence; it does not compare coverage names with the frozen master list
   or require a nonempty frame/design library.  M2148 could catch this later,
   but M2147's own raw-PASS parser is not fail-closed against these mutations.
2. **The process JSON is not internally validated.**  A fixture claiming
   `unique_process_identity_count=2` and `root_seen=true` while listing zero
   processes is accepted.  Parent relationships, executable identities, one
   top-level ICC2 process, and count/list agreement are unchecked.
3. **The monitor's conversion-child classifier omits installed executable
   names.**  It does not recognize `icc2_exec`, `dgcom_exec`, or
   `lm_shell_exec`; because an `exec` can retain PID/start-time and overwrite
   the previously sampled wrapper record, the reported conversion-child count
   can be zero even when a conversion executable was observed.
4. **The repository-root snapshot sees only regular top-level files.**  A new
   top-level directory or symlink escapes comparison, contrary to the contract
   wording that root collateral is identical before and after invocation.
5. **The author receipt is not currently exhaustive.**  Its listed hashes and
   outer seal verify, but two regular files are present outside `SHA256SUMS`:
   `__pycache__/selfcheck.cpython-312.pyc` and
   `__pycache__/tests.cpython-312.pyc`.  They are benign Python cache files, but
   the directory cannot be described as exhaustively sealed in its present
   state.

## Three P2 hardening items

- The wrapper script is pinned, but the currently selected real executable
  `/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/dgcom_exec` (SHA256
  `4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c`)
  is absent from the contract and runtime identity checks.
- The core-site gate falls back from exact `core` to wildcard `*core*`; the
  contract requires a valid `core` site, so the fallback is weaker than the
  stated predicate.
- An isolated `HOME` should be created and exported even after adding
  `-no_init`, so any tool sidecar or secondary process has no route to the user
  home.

## Minimum safe successor

Do not modify or retry M2135/M2141/M2147.  A new-numbered additive source can be
small:

1. invoke exact ICC2 as `-no_init -f`, set `HOME`, `TMPDIR`, cache, conversion
   output, logs, and design library under one isolated directory, and pin the
   resolved real executable as well as the wrapper;
2. require exact `core`, strengthen the root inventory to include every
   top-level node type, and keep the existing absorption of the exact M2135
   collateral;
3. make the process census self-consistent and classify the actual installed
   ICC2/LM conversion executable names;
4. require exact gate tokens and values, compare coverage names to the frozen
   94-master file, reject empty frame/design-library outputs, and validate the
   process list rather than trusting summary counters;
5. run a new independent source hammer and double seal before consuming one
   fresh license query and one fresh ICC2 preflight.

Only a successor with score at least 95 and P0/P1/P2 = 0/0/0 may authorize one
library-import-only execution.  This review authorizes zero license queries,
zero ICC2 sessions, and zero P&R runs.

## Score and boundary

- Score: **82/100**
- P0/P1/P2: **1/5/3**
- M2147 authorized: **false**
- library import proven: **false**
- RTL/P&R/timing/area/power evidence: **false**
- paper-PPA-ready: **false**
