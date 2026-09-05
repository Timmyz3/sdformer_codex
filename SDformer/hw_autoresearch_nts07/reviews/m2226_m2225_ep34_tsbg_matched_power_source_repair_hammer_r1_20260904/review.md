# M2226 independent source repair review

**FAIL, 92/100; P0/P1/P2 = 1/0/0. M2227 is not authorized.**

M2225 correctly pins M2172 and M2117, rejects their drift before contract validation, binds both in the future review and result, and fixes the SSG/FFG mapping versus TT PTPX corner description. All 28 source hashes match, the original M2217 inventory is unchanged, and the three selected rows and fixed weights reproduce from the 2,880-row population. The matched SRAM numbers and serial tool budget are unchanged.

The repair misses one further executable dependency:

```text
M2217 parser
  +-- M2172 helper (pinned)
  |     +-- M2160 helper (unconditionally imported, unpinned)
  +-- M2117 helper (pinned)
```

M2172 executes `BASE = load_base()` on import. That loader executes `parse_m2160_m2018_ordinary_native_saif_report_reset_preflight.py`, whose SHA is `381fbaac6c75aed86aa1dd12aad41dffeb8348c7a875e95f1c162256df6ba22b`. Neither the runner inventory nor a helper-local gate checks it. M2160 and M2117 import only standard-library modules, so this is the last project-local Python dependency found in the recursive inspection.

The independent test traced the real runtime imports, confirmed that the source gate never hashes M2160, and replaced only its import loader in memory. Source validation still passed, followed by execution of the injected M2160 marker. This reproduces the identity hole without changing any reviewed file. M2217 currently does not call M2160's `audit_single_axis_source`, so this finding establishes an uncontrolled import-time execution path; it does not claim that existing power arithmetic has been falsified.

The fix is small: bind the complete three-helper closure in the successor inventory, preflight, review and result identities; exercise mutation rejection for every member. Preserve M2225/M2227 and request a new independent source review. The expected campaign remains 1 license query, 2 VCS compiles, 6 simulations, 6 diagnostic and 6 measurement SAIFs, 2 DC maps and 6 PTPX points, all serial and without automatic retry.

Validation: author's 8 tests pass; 16 independent investigative checks pass and reproduce the P0. No EDA, license, GPU, Git mutation or production-source edit occurred. `docs/359` retains its frozen SHA. The reusable M2217 Tcl/internal marker text still names the old M2219 lineage, but all actual output paths and the outer final result use fresh M2227 identity; no old attempt is read or consumed.

Tool provenance was checked read-only: versioned VCS/PT/LMUTIL paths are regular files; the DC launcher symlink resolves to the exact pinned `snps_shell` SHA. This review creates no physical or power result.
