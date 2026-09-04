# M523 r2/r3 minimum-repair author handoff

## Outcome

The sealed exit-31 diagnosis was implemented exactly. The TB has one new line, `@(negedge clk_core);`, immediately after the drain loop and before final checks and `$finish`. Removing this exact line reconstructs the failed-run TB SHA `1769ced4...bc31`; the repaired TB SHA is `3b468b72...f5cc`.

RTL, SVA, and filelist are byte-identical to the failed run. No functional ledger, architecture, cover definition, or claim was changed. No EDA tool was run by the author.

## New identities

- Runner r2: `60f85b104839cd8d340ba7592147117fc930d46742784c644a385a2ed470ece6`
- Contract r3: `6dac33f9fe035c0ed1c14ddd7dbc7d9ebfabcdec279cf027ce07cf0774baa415`
- TB r3: `3b468b7247ddbb0f292a653ba15f0021ca5926354128858208ccf6147d6ff5cc`
- Failure-review outer seal file: `b3c2ec802dc053c84e1154369ee23045724f303585a9ebea3260022b6b96b0ad`
- Static-review request: `80aa357880f06731e451bf043452b44fb1cfa91819115293292b1566717c505e`

Runner r2 uses new canonical, attempt, work, quarantine, wrong-runner receipt, functional receipt, topology, and symlink-inventory identities. It hard-gates both the sealed failure diagnosis and the future independent r2 static review. The consumed r1 attempt and r1 quarantine are neither reused nor promoted.

## Static-only checks

`bash -n`, seven embedded-Python compilations under the installed Python 3.6 parser, strict finite contract parsing, the exact-one-line TB reconstruction, and the independent 43-tap integer oracle all pass. The topology verifier uses Python-3.6-compatible `relative_to`/`ValueError` containment checks instead of unavailable `Path.is_relative_to`. The oracle remains fanouts `[4,6,6,9,9,9]`, bundles `[8,2,6,8,1,8,8,2]`, and phase totals `[6,10,10,17]`.

## Claim boundary and next gate

All direct-C2, performance, energy, area, timing, PPA, system, and headline fields remain false. This author handoff does not authorize VCS. A different independent reviewer must inspect the frozen request, create the exact r2 static-review topology, and authorize at most one invocation of the exact runner SHA.
