# M2234 independent review request

Review only the additive M2233 source repair. The author previously completed M2226's independent FAIL review, then authored M2233. This successor requires a different independent reviewer.

No EDA, license, GPU or Git action is authorized by this request. Use CPU-only source checks and mutation tests. Keep all old M2217/M2225 sources, old result namespaces and `docs/359` unchanged.

The complete project-local parser import graph is:

```text
parse_m2217...py -> parse_m2172...py -> parse_m2160...py
               -> parse_m2117...py
```

Check all three helper hashes, recursive closure, 29 source inventory entries, preflight ordering, review/result helper identity binding, fresh M2235 namespace and fixed budget. M2160 and M2117 have only standard-library imports. Verify the M2233 tests reproduce every helper drift, missing/wrong review helper binding and score-94 rejection.

Check the unchanged SSG0P9V125C slow/max plus FFG1P05VM40C fast/min mapping and TT0P9V25C PTPX, 22.213 pJ per accepted bank activation and 3.826774326764422 mW SRAM leakage proxy. The result must explicitly label these as mixed-corner component modeling. All six points are required. Selection remains frozen; its max-request tie-break means the fixed one-third aggregate is a **three-window weighted index**, not a population mean or frame energy.

Expected review directory:
`reviews/m2234_m2233_ep34_tsbg_matched_power_source_repair_hammer_r1_20260905`

If and only if score is at least 95 and P0/P1/P2 are all zero, use:

```json
{
  "status": "PASS_M2234_M2233_MATCHED_POWER_SOURCE_REPAIR_RELEASE",
  "score_over_100": 95,
  "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
  "authorization": {
    "license_queries": 1, "vcs_compiles": 2, "simv_runs": 6,
    "diagnostic_saif_files": 6, "measurement_saif_files": 6,
    "dc_runs": 2, "ptpx_runs": 6, "automatic_retry": false,
    "p1_serial": true, "reuse_m2203_raw": false
  },
  "identity": {
    "runner_sha256": "fb4610913b2f60e3c1443e42441e00361adb5f57b419a2aa2e366ee98fe03857",
    "contract_sha256": "3a4d9981f084e2f4b08ff3ca1685ffcca55b18b6615647515f358ebab86ad391",
    "m2172_helper_sha256": "42fd87d6991c46366e80db1d08c20ec5e0d463f3bca8c6050673093d04f3bfe2",
    "m2117_helper_sha256": "2787e8858799577db8f87297d2d1c1c16ccf0a3933b00f9a039071e968ea3547",
    "m2160_helper_sha256": "381fbaac6c75aed86aa1dd12aad41dffeb8348c7a875e95f1c162256df6ba22b"
  }
}
```

The actual earned score may exceed 95. Include all review files in an exhaustive `SHA256SUMS` and seal that manifest. A passing source review authorizes only the sole M2235 campaign; M2236 independent result review is still required before any paper claim.

Run command after release, by the root agent only:

```sh
/opt/anaconda3/bin/python3 hw_autoresearch_nts07/dc_handoff/scripts/run_m2233_ep34_tsbg_matched_power_repair_one_shot.py
```
