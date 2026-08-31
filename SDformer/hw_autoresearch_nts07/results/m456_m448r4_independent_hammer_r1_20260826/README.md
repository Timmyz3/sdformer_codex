# M456 independent hammer of M448R4 PTPX

This directory independently validates the sealed M448R4 selected-slice prelayout standard-cell PrimeTime PX run without modifying the R4 directory or reusing obsolete R1/R2/R3 numeric power.

- `m456_independent_recomputation.json`: machine-readable independent manifest, SAIF, power, energy, sensitivity, and boundary audit.
- `m456_power_reparse.csv`: raw-report reparse of 50/100/200 ps power components.
- `r4_manifest_self_check_from_r4_cwd.log`: 44-entry manifest self-check executed from the R4 cwd.
- `r4_outer_seal_self_check_from_r4_cwd.log`: outer seal self-check executed from the R4 cwd.
- `m456_m448r4_independent_hammer_review.json`: scored P0/P1/P2 admission record.
- `m456_m448r4_independent_hammer_review.md`: human-readable review.
- `M456_REVIEW_SHA256SUMS`: inner evidence manifest.
- `M456_REVIEW_SHA256SUMS.seal.sha256`: outer seal of the inner manifest.

Admitted primary point: 6.25380802 mW and 18.76142256815862 pJ per measured cycle at 100 ps input slew, strictly within the frozen M416 selected-slice prelayout standard-cell boundary.

