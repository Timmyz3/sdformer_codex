# M2136 independent failure hammer of M2135 matched ICC2 P&R

## Verdict

**PASS failure diagnosis; M2135 is consumed, non-retriable, and not citable.**
The sealed M2135 quarantine contains one healthy license preflight and one
`ordinary_lru4` ICC2 transcript.  No `tsbg_b4` ICC2 transcript or axis directory
exists.  The ordinary axis exits with code 42 during its first `create_lib`;
the report, output, raw-parasitic, and library-cache directories are empty.
There is no linked design, placement, CTS, route, timing, area, power, DEF, or
SPEF result.  Consequently M2135 supplies **zero** physical/PPA evidence.

The attempt marker records the authorized budget (`license_queries=1`,
`icc2_shell_runs=2`), not observed execution counts.  The observed census is
one license query, one ordinary-axis ICC2 invocation, and zero TSBG-axis ICC2
invocations.  The runner is sequential and fail-closed, so the ordinary-axis
nonzero exit prevents entry into the second loop iteration.

The attempt and quarantine directories each pass their directory-local,
exhaustive inner manifest and outer manifest seal.  However, ICC2 also wrote an
untracked transcript sidecar at repository root, `icc2_output.txt` (25,324
bytes, 472 lines, SHA256
`0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6`).
It contains the same CMD-104/LIB-117/FILE-001/LIB-027 failure chain but lacks
the Tcl catch stack found in the sealed ordinary log.  It is outside the
quarantine manifest.  This is a **P1 evidence-boundary defect**: the quarantine
seal is valid for its directory but is not exhaustive over all process output.
M2136 does not modify, delete, or silently absorb that collateral.

## Exact observed failure chain

The first runtime diagnostic is at sealed log line 465:

1. `set_app_var lib.configuration.local_output_dir ...` produces CMD-104: the
   name is not an application variable and is treated as a Tcl global.
2. `create_lib` then emits LIB-117: library configuration is skipped because a
   technology file was not specified.
3. ICC2 attempts to open `<Milkyway-directory>/lib.ndm`, which does not exist,
   and emits FILE-001.
4. The same physical-source directory is rejected as a valid reference library
   with LIB-027.
5. The M2133 catch wrapper reports `problem in create_lib` and exits 42.

This interpretation is supported directly by the installed V-2023.12-SP3
command reference:

- `set_app_options -name ... -value ...` is the setter for application options.
- `lib.configuration.local_output_dir` controls where converted cell/frame
  libraries are written when Milkyway physical source data is used.
- `create_lib` accepts Milkyway libraries as physical source data and exposes
  an explicit `-technology` input.
- LIB-027 states that without a technology file, physical sources are treated
  as full NDM references, which explains the attempted `lib.ndm` open.
- `generate_frame_from_mw` is the documented command for creating a frame NDM
  from a Milkyway FRAM library.

The four runtime diagnostic codes each occur exactly once after the echoed Tcl.
This is the unique observed first-cause chain.  It does not prove that no later
library, layer, linking, floorplan, timing, or routing failure will appear after
repair; those stages were never reached.

## Physical-source audit: do not guess a technology file

The frozen Milkyway source remains the exact 1,051-file inventory identified by
manifest SHA256
`7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3`.
It contains 1,044 FRAM cells, two CEL records, and its Milkyway library metadata.
That is the only physical source currently pinned by the M2133 chain.

Two nearby files must **not** be promoted to a usable ICC2 technology source
without a preflight:

- The 9.2 MB standard-cell LEF, SHA256
  `c1d0f2ad27efe7d6ac3ba74236ca9dfe63e51d4624242282f6b3237f5a0e6df2`,
  is a macro-geometry LEF.  It contains macro references to `SITE core` but no
  technology `SITE` or routing `LAYER` declarations.
- The PDK path named `Techfile/online/1P9M_6X1Z1U` contains an OA/Virtuoso
  technology database; its top `techfile` SHA256 is
  `9cd1073ab64920e056c8cc7319651866a47072de7e5e824fa792471f4552d848`.
  Its presence and stack name do not establish that it is a valid Synopsys
  `create_lib -technology` input.

Therefore the preferred next route is the documented, explicit
`generate_frame_from_mw` preconversion of the frozen Milkyway inventory into a
new isolated frame NDM.  An alternative direct `create_lib -technology ...`
route is permitted only after an exact compatible Synopsys 1P9M/6X1Z1U
technology file and complementary physical LEF are identified, hashed, and
source-reviewed.  M2136 does not guess that identity and does not claim either
route already works.

## Minimum safe successor sequence

M2133/M2135 remain immutable.  A new-numbered source must first perform a
**library-import-only preflight**, not another full P&R:

1. Set the option with the documented form
   `set_app_options -name lib.configuration.local_output_dir -value <isolated-cache>`
   and require
   `get_app_option_value -name lib.configuration.local_output_dir` to return
   that exact isolated path before conversion.
2. In a fresh, axis-independent working directory, invoke the documented
   `generate_frame_from_mw <leaf>.ndm -mw_lib <frozen-MW> -log_file_dir ...
   -output_directory ...`.  Require return status 1, a regular nonsymlink NDM,
   a complete output inventory, and an exhaustive seal.  Do not use `-overwrite`.
3. Create a disposable design library from that frame NDM and the exact frozen
   TT/SS/FF DBs.  Before any RTL import, require the expected physical-cell
   inventory, the 94 mapped master names covered in all logical corners and the
   physical frame, a valid `core` site, exact M1--M9/VIA coverage, and an
   explicit 1P9M/6X1Z1U technology identity compatible with the frozen NXTGRD
   and layer map.
4. Run from the isolated preflight directory, never repository root.  Snapshot
   and compare the preflight cwd and repository root before/after execution;
   any `icc2_output.txt` or other collateral must be moved into the failure or
   result package before sealing.  Any unexpected external write fails closed.

The preflight budget is exactly one license query and one top-level ICC2
invocation, with zero placement/CTS/route runs and no automatic retry.  Any
tool-spawned conversion child must be logged and counted.  A new independent
source hammer must pin the option syntax, official-document hashes, Milkyway
manifest, conversion command, expected output inventory, cwd isolation, and
budget before that single preflight is authorized.  Only a separately sealed
successful preflight may authorize a later matched two-axis full-P&R source.

## Severity and claim boundary

- P0: 0
- P1: 1 — untracked repository-root `icc2_output.txt` escaped the otherwise
  valid quarantine seal.
- P2: 0

M2135 is not a post-route result, not a timing result, not a placement/routing
comparison, not a power result, not macro-inclusive, and not paper-PPA-ready.
It must not contribute a number or success statement to TCAS-II/ISCAS tables.
No EDA executable, license utility, or GPU process was invoked by M2136, and
the protected `docs/359` SHA remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
