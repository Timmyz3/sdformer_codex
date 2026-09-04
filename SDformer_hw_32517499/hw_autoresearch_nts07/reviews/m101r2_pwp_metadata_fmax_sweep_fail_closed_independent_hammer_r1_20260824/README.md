# M101-r2 PWP metadata Fmax sweep independent hammer review

Status: **conditional pass**. The frozen 16-point Synopsys DC evidence and its headline-safe numeric result reproduce, but the r2 post-run auditor is not yet fully fail-closed under upgraded hostile fixtures.

## Independently reproduced result

- Reparsed all 16 point directories directly: design top, requested and reported clocks, setup/hold slack and status, QoR WNS/TNS/path counts, constraints, DC backend identity, mapped Verilog/SDC/DDC and SVF presence.
- Rechecked frozen RTL, filelist, functional contract, VCS receipt, DC Tcl/SDC and setup/hold library SHA-256 identities.
- M85 first passing frozen-grid target: 4.000 ns, setup `MET`, WNS 0.0000 ns, TNS 0, area 28,928.718041 um2.
- M99 first passing frozen-grid target: 2.750 ns, setup `MET`, WNS +0.0009 ns, TNS 0, area 13,832.784035 um2.
- Same-recipe logic-only pre-macro frozen-grid closure ratio: 1.454545x. Fastest-point mapped standard-cell area fraction: 0.478168x, or 52.1832% lower.
- Exact-zero boundary was checked explicitly: `MET` with zero slack passes; `VIOLATED` with zero slack does not.

These are frozen-grid target-closure and mapped-cell-area results. They are not continuous/post-layout Fmax, module or system throughput speedup, macro-inclusive PPA, power, or energy.

## Manifest and receipt audit

- The durable manifest has 374 unique, currently safe labels: 370 run files (16 x 23 point files plus 2 top-level files), the original contract, r2 auditor, setup library, and hold library.
- All 374 current digests verify.
- Each of 16 receipt points exposes seven hashes (identity, QoR, setup, hold, mapped Verilog, mapped SDC and DDC); all 112 agree with the durable manifest.
- The receipt binds the durable-manifest SHA/count. The five-entry complete manifest binds the original and r2 seal contracts, auditor, receipt and durable manifest.
- A nominal r2 auditor replay regenerates both the receipt and 374-entry durable manifest byte-for-byte.

## Hostile result and score

Score: **83/100**; P0: **0**, P1: **3**, P2: **5**.

Nine attack classes fail closed without producing a receipt: symlink and copied-point aliases caught by identity/clock checks, missing or empty mapped Verilog, contract-threshold drift, setup-library replacement, filelist replacement, and RTL replacement.

Five upgraded attacks remain fail-open:

1. A copied 3.000 ns point with forged identity and clocks is accepted as 2.750 ns even though mapped SDC still requests 3 ns.
2. Tiny nonempty fake mapped Verilog/SDC/DDC/SVF files are treated as authentic mapped artifacts.
3. Deleting an existing non-required precompile report is accepted and silently shrinks the manifest to 373 entries.
4. A newline-containing filename injects a duplicate physical manifest label while the audit still passes.
5. Negative QoR TNS contradicting a positive `MET` timing report silently downgrades the point instead of invalidating the evidence set.

The production evidence itself is not shown corrupt; these attacks expose resealing weaknesses. Required P1 repairs are backend-bound point provenance including mapped-SDC period, real artifact validation/reopening or equivalence, and a fixed canonical 374-label inventory rejecting control characters and duplicate normalized paths.

## Reproduce this review

From `hw_autoresearch_nts07`:

```bash
python3 reviews/m101r2_pwp_metadata_fmax_sweep_fail_closed_independent_hammer_r1_20260824/audit_m101r2_fail_closed_independent_hammer.py
sha256sum -c reviews/m101r2_pwp_metadata_fmax_sweep_fail_closed_independent_hammer_r1_20260824/input_manifest.sha256
(cd reviews/m101r2_pwp_metadata_fmax_sweep_fail_closed_independent_hammer_r1_20260824 && sha256sum -c manifest.sha256)
```

The audit creates hostile fixtures only inside temporary directories and writes only its machine-readable output in this review directory. It does not modify production sources or `docs/359_DATE终局冻结_20260813.md`.
