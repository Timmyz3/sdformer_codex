# M629 fresh independent hammer of M628 registry r2

## Verdict

**NO_GO — registry methodology needs revision; no Table A row or paper headline is admitted.**

- Score: **68/100**
- Severity: **P0=0, P1=4, P2=1**
- Canonical state: 12 frozen sources valid, 7 Table A rows present, 0 eligible rows, headline false, analytical range false.
- Review scope: registry soundness and the concrete no-EDA queue only. This review has no authority to admit a speedup, system-energy, PPA, accuracy, or Strong-Accept headline.

The current canonical file is conservative, but the executable gate is not fail-closed under coordinated mutations. A future editor can relabel the official external M618 source as six `DIRECT_UNIFIED_CYCLE_SIM` Table A rows, self-assert all closure booleans, insert fake receipt strings and multiply local ratios into cycles; the builder returns `headline_admitted=true`. This violates the central M629 pass boundary.

## Blocking findings

### M629-P1-01 — Table A evidence and completion receipts are self-attested

The gate checks row strings, finite numbers and booleans, but it does not parse the cited result source and reconcile its schema, measurement class, population, resource manifest, completion receipt, cycles, energy, area, accuracy or independent-hammer status. It also does not bind `completion_receipt_sha256` to a repo-confined regular file.

Independent attack: all six mandatory rows cited `m618`, which is explicitly an official external Prosperity Table C result. The attack set arbitrary positive measurements, fake 64-character receipt/resource strings, all closure booleans, three sequence labels and the five aggregate/view labels. `ours_exact.cycles` was derived by multiplying the M528 and M618 local ratios. After synchronizing the self-reported claim count, the builder admitted a **4.144851634587013x** headline while `ours_lossy` remained unfilled.

Required fix: Table A sources need a dedicated, immutable result schema and allowlist; every numeric/identity/closure field must be recomputed from or exactly matched to a repo-confined, SHA-bound result and independent receipt. Table C source IDs must be structurally forbidden in Table A. Aggregate values, sequence receipts and density preregistration need evidence bindings, not only presence strings. Ratio provenance must be direct cycles from the bound result.

### M629-P1-02 — the six-row denominator ladder is config-controlled

Deleting `exact_bit_k1x8` from both `rows` and `required_row_ids` is accepted. The builder never compares the list to a code-level mandatory set and never consumes `strongest_same_page_baseline_row_id` in the gate.

Required fix: hard-code and test the exact six IDs, their roles and exact fidelity; hard-code the Dense96 numerator, K1x8 strongest same-page baseline and `ours_exact` candidate, or bind them to an immutable contract that the builder itself verifies. Add coordinated deletion/renaming attacks.

### M629-P1-03 — strict JSON parsing does not cover evidence sources

`load_json()` rejects duplicate keys and NaN/Infinity in the registry config, but `validate_sources()` only hashes source bytes. A repo-local JSON source containing `{"metric":1,"metric":2}` was SHA-bound and accepted as a thirteenth source.

Required fix: strict-parse every JSON evidence source after path and SHA validation. If non-JSON source types are later needed, register explicit source media types and validators.

### M629-P1-04 — M518 Table B provenance is misbound

Registry row `m518_fixed_t10_directed_issue_cycle_anchor` labels 17 cycles as `DIRECTED_RTL`, but source `m518` is the pre-run static hammer. That source says `canonical_result_absent_at_review=true`, `rtl_cycles_admitted=false`, and VCS behavior/cycle observations remain unadmitted until a result hammer. The actual post-run receipt hammer exists separately at `reviews/m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_r1_20260827/...`, but it is not a frozen M628 source.

Required fix: bind the Table B row to the actual post-run receipt hammer and exact SHA, or downgrade it to a static source-identity expectation rather than a directed RTL measurement.

## Non-blocking finding

### M629-P2-01 — rounded values lack a registry-wide precision rule

M519 stores `[1.012, 1.0392]` while direct cycle division gives `[1.012185215272136..., 1.039215686274509...]`. M618 and M526 values are also intentionally shortened. The values are directionally correct, but each row should carry either exact Decimal strings or an explicit rounding/precision field.

## Independent numeric and provenance checks

| Registry evidence | Independent result | Finding |
|---|---:|---|
| M528 vs M468 strong-zero | `760350133 / 435293339 = 1.74675343010475058...x` | Number/scope correct; local four-Conv, one-sequence CPU only |
| M528 vs same-coordinate bit | `757946784 / 435293339 = 1.74123221306632491...x` | Number/scope correct; M573 independently closes this waterfall value |
| M623 component energy | `(3.3018888289181215 - 2.039632600258816191) / 3.3018888289181215 = 38.228307918921944...%` | Correct; generated nine-macro component only |
| M481 FC1 lifecycle | `9292511340 / 4603346064 = 2.01864278957238093...x` | Correct analytical local point; no physical SRAM/full FC1 |
| M481 envelope sensitivity | `620302905 / (620302905-100895624+49981910.87655147) = 1.08941812357844520...x` | Correct sensitivity, not measured speedup |
| M480 BN | `61568856 / 41048856 = 1.49989212854068332...x` | Correct exact-schedule CPU model; no runtime-affine RTL/macro |
| M518 ATLIF | 17 cycles appears only as a preserved pre-run source expectation in the bound source | **Provenance failure; see P1-04** |
| M519 K8/K1x8 | exact nonempty-case range `1.012185215272136...–1.039215686274509...x` | Correct directed RTL range, rounded in registry |
| M523 bundler | 43 taps, 8 bundles | Correct descriptor-only directed function |
| M618/M619 official FC1 | `757894814 / 319397528 = 2.3728887907986564...x` | Correct external artifact, 100 records, not ours/full network |
| M526 Prosperity workbook | geomean `7.313884876012107x` | Registry rounding correct; cross-paper context only |
| M532 Phi/FireFly-T | `reported_only`, `ours=false`, `headline_eligible=false` | Correct literature-only boundary |

Independent analytical range:

- `1442206883 / 803774000 = 1.7942940217026179000564835389...`
- `1442206883 / 790920000 = 1.8234548159105851413543721236...`

The arithmetic is correct and remains non-admitted. It lacks a direct unified decoder-complete run, timed memory, common resources and completion receipts, population coverage, logic/SRAM/DRAM energy, logic+macro area/STA and an independent result hammer.

## Table and headline policy audit

- Canonical Table A contains exactly the six intended mandatory rows plus optional `ours_lossy`; all seven are incomplete and zero are eligible.
- Optional `ours_lossy` does not block an otherwise eligible exact ladder. This behavior is correct, although the synthetic test also exposes P1-01.
- Table B and C row flags reject simple promotion; Table C also rejects `ours=true`. Coordinated B/C-to-A relabelling is not rejected.
- Partial Table A cycle filling with decoder/memory/energy/PPA/hammer fields missing remains blocked.
- The policy names arithmetic mean, geometric mean, ratio of sums, minimum, maximum, default geomean, iso-lane, iso-service and either three sequences or low/mid/high density strata. The current implementation checks only strings/counts, not their bound measurements or preregistration.

## E1–E4 and no-EDA queue

E1–E4 are a credible minimal **conditional** evidence plan: exact decoder/population closure; one direct unified cycle+memory driver across B0/B1/K1/K1x8/K8/Ours; later matched logic+macro/STA and logic/SRAM/DRAM energy; and multi-sequence or preregistered density coverage followed by a blind hammer. They do not guarantee a 3.8/5 score and do not require a fourth matcher.

The seven no-EDA tasks have concrete outputs and mostly correct boundaries. N2 remains Table C; N3/N4 remain Table B until unified; N5 supplies population coverage; N6 is the Table A adapter; N7 refuses premature component-energy promotion. They may proceed after the registry source/gate repairs, but their outputs cannot be admitted through the current gate.

## Mandatory check disposition

| Check | Result |
|---|---|
| H01 frozen SHA, config strict JSON, path checks | Partial: hashes/config/path pass; source-JSON strict parse fails |
| H02 all 12 sources and B/C values | Fail: M518 provenance misbound; all other listed values/boundaries verified |
| H03 Decimal analytical recompute | Pass |
| H04 analytical non-admission | Pass |
| H05 six mandatory rows + optional lossy, zero canonical eligible | Canonical pass; coordinated deletion attack fails |
| H06 no B/C promotion/external-as-ours | Fail under coordinated Table A relabelling |
| H07 direct unified identity/energy/PPA/accuracy/hammer gate | Fail: self-attested, not evidence-bound |
| H08 aggregates/views/coverage | Policy present; fail as evidence gate because labels alone suffice |
| H09 E1–E4 conditional path, no fourth matcher | Pass as planning statement only |
| H10 concrete no-EDA queue and promotion boundaries | Pass |
| H11 11 CPU tests and canonical builder | Pass, but suite misses the three blocking attacks |
| H12 docs359/no EDA/GPU/paper-body | Pass |

## Execution boundary and protected identity

- Python: 3.6.8; CPU-only.
- Unit tests: 11/11 pass.
- Canonical builder: `M628_REGISTRY_PASS sources=12 table_a_eligible=0 headline_admitted=false analytical_admitted=false`.
- No GPU, EDA, remote, production simulator or paper-body task was run.
- docs359 before/after SHA-256: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Re-review gate

M630-style repair should (1) hard-code the ladder and anchors, (2) strict-parse every JSON source, (3) bind Table A numeric/identity/closure fields to dedicated direct-result and independent-hammer schemas, (4) forbid Table C source IDs in Table A, (5) bind real aggregate/coverage receipts, (6) correct the M518 source, and (7) add the three failing attacks to the unit suite. Only a fresh independent hammer with P0=0/P1=0 may admit the registry methodology; no repair alone admits a paper headline.
