# M1286 — M1281 decoder surrogate receipt-blind source hammer

## Verdict

`STOP_M1281_REAL_ADAPTER_AND_ANALYTICAL_ANNEX__SOURCE_REPAIR_REQUIRED`

Score: **61/100**.  M1281 is a useful synthetic calibration scaffold, but it is
not safe to connect to a sealed M1111DR2 result yet.  The strongest checks work:
exact 120-row cardinality, ordinal/layer ordering, exact JSON key sets,
single-field traffic conservation, claim-boundary equality, and the inclusive
0.1% maximum-relative-error gate all fail closed as intended.  Synthetic output
also remains forbidden from the analytical annex in the ordinary Boolean path.

Four admission blockers remain:

1. A non-fixture payload with arbitrary 64-hex strings and two self-declared
   PASS booleans is accepted as sealed/hammered and can receive analytical-annex
   eligibility.  No seal or hammer artifact is opened or cryptographically
   verified by `calibrate_payload`.
2. `synthetic_fixture` is compared by object identity but never required to be
   Boolean.  Passing integer `0` in both the authority and function argument
   enters the non-fixture branch and enables the annex.
3. `commit_bytes = 288 per call` is incompatible with M1111DR2 transaction
   semantics.  M1111DR2 commits 288 bytes for every timestep, destination and
   96-channel output block: D0/D1/D2/D3 are respectively
   13,824,000 / 27,648,000 / 55,296,000 / 221,184,000 bytes per call.
4. Group, term, traffic and cycle fields are mutually self-consistent but not
   bound to M1111DR2 kind summaries or digests.  A coordinated forged row is
   accepted, including a positive group count with zero active terms.  A nominal
   120-row payload with only one distinct observation per layer is also accepted.

Consequently this milestone admits only the **calibration framework source**.
It admits no real calibration, analytical-cycle annex, decoder cycles, traffic,
speedup, system speedup, PPA, energy or paper number.

## Receipt-blind scope

The hammer read only these five source/contract files:

| File | SHA256 |
|---|---|
| `system_simulator/scripts/build_m1281_decoder_cycle_traffic_surrogate_calibration_source.py` | `098d7c0e96df18ed9eda2f43e26230b86ba5afbef3975c46d695ec8953e7a4ce` |
| `system_simulator/tests/test_m1281_decoder_cycle_traffic_surrogate_calibration_source.py` | `c812b11c05d4fc00b30b4d029686e0d245aaefafb27ca1135c11fca78c14f170` |
| `contracts/m1281_decoder_cycle_traffic_surrogate_calibration_source_contract_r1_20260830.json` | `829a0766f1d79a8acfdade0fd42853f445699e533b9ab918c745e8bc460501f9` |
| `system_simulator/scripts/run_m1111dr2_m1105dr2_decoder_only_production_zero_arg.py` | `1167258c228631b73ca1784ae57db19e8f0fbe709efa34f369585c508bc9d746` |
| `contracts/m1111dr2_m1105dr2_decoder_only_production_runner_source_contract_r2_20260830.json` | `821819b00503b91a8fb8dfca8fe000208e10746e751a3815131dc8ff1cbed515` |

It did not read the M1281 author receipt, its test receipt, a live work prefix,
any M1111DR2 result, or any canonical calibration payload.  It ran only
in-memory synthetic tests.  It did not run real calibration, EDA, GPU or remote
work.

## Attack matrix

| Attack | Observed result | Severity / interpretation |
|---|---|---|
| Exact synthetic baseline | PASS, annex false | Correct |
| Arbitrary 64-hex result/seal/hammer identities plus PASS booleans, non-fixture mode | **Accepted; annex true** | Critical: future adapter can bypass seal verification |
| `synthetic_fixture=0` in authority and call | **Accepted; annex true** | High: Boolean/type confusion |
| 119 rows | Rejected by existing self-test | Correct |
| Global ordinal mutation | Rejected | Correct |
| Layer order mutation | Rejected | Correct |
| One traffic field changed without its drivers | Rejected | Correct |
| Coordinated group/term/traffic/cycle forgery | **Accepted** | High: all fields are self-reported; no source binding |
| Positive group count with zero active terms | **Accepted** | High: violates nonempty bank-unique group semantics |
| 120 nominal rows but one unique sample per layer | **Accepted** | High: no effective-sample/sequence-stratum gate |
| Exact max relative error = 0.001 | Accepted | Correct inclusive boundary |
| Max relative error > 0.001 | Rejected | Correct |
| Input `system_speedup_admitted=true` | Rejected | Correct |

The pseudo-seal and Boolean attacks are not hypothetical adapter mistakes: the
public import target is `calibrate_payload`, and that function itself treats
formatted SHA strings and Boolean assertions as authority.  A docstring asking
a future adapter to verify seals is not an enforceable trust boundary.

## M1111DR2 semantic compatibility

Five traffic expressions are compatible with M1111DR2 only when their drivers
are derived from exact transaction summaries rather than copied from a new
self-reported schema:

- descriptor bytes = 16 x `input_descriptor_read.count`;
- weight bytes = 16 x active source terms, with active terms derivable from
  `weight_read.traffic_bytes / 16`;
- psum read bytes = 6 banks x 48 B x group count = 288 x groups;
- compute count = group count;
- psum write bytes = 6 banks x 48 B x group count = 288 x groups.

The commit expression is incompatible.  M1111DR2 loops over ten timesteps,
every output destination and every `ceil(Cout/96)` output block.  Therefore:

| Layer | Correct M1111DR2 commit bytes/call | M1281 value | Under-count |
|---|---:|---:|---:|
| D0 | 13,824,000 | 288 | 48,000x |
| D1 | 27,648,000 | 288 | 96,000x |
| D2 | 55,296,000 | 288 | 192,000x |
| D3 | 221,184,000 | 288 | 768,000x |

The cycle equation `4 * group_count + layer_constant` is not a transaction
identity.  The M1111DR2 scheduler includes dependency completion, port
contention, outstanding limits and cross-call scheduler state.  It is a
calibration hypothesis and may enter an analytical annex only after a sealed,
stratified validation demonstrates the frozen error gate.  This hammer neither
confirms nor rejects that empirical fit.

## Score

| Dimension | Score | Reason |
|---|---:|---|
| Scope isolation | 17/20 | CLI is synthetic-only and ordinary fixture output cannot annex |
| Schema/order/type safety | 13/20 | Exact keys and order are strong; Boolean type and semantic-domain checks are missing |
| Authority/seal integrity | 3/20 | Format-only hashes and self-declared PASS values can authorize annex |
| M1111DR2 semantic fidelity | 11/20 | Five component formulas align conditionally; commit is wrong by 48,000x--768,000x |
| Statistical/claim discipline | 17/20 | Error and claim gates work; no effective-sample or held-out stratum requirement |
| **Total** | **61/100** | Repairable framework; real adapter and annex remain STOP |

## Required actions

### P0 — mandatory before any real adapter or annex

1. Split fixture and production entry points.  Require
   `type(synthetic_fixture) is bool`; preferably remove the caller-controlled
   flag entirely from the production function.
2. Make the production adapter verify a supplied sealed directory itself:
   regular non-symlink files, exact top-level set, manifest coverage and member
   hashes, outer seal, exact result schema/status, and the independently sealed
   result-hammer identity.  Do not accept `*_pass` booleans or naked SHA strings
   as authority.
3. Project directly from exact M1111DR2 call rows.  Bind sequence ordinal/name,
   sample ID, module ordinal/name, configuration, three transaction digests,
   measured diagnostic cycles and all six kind summaries.  Derive rather than
   accept group count, active terms and traffic.
4. Replace fixed commit bytes with the exact M1111DR2 `output_commit` traffic.
   Require `group_count <= active_source_terms <= 8*group_count` for nonempty
   groups and equality of descriptor/compute/psum-read/psum-write group counts.
5. Add regression attacks for forged authority, integer/None fixture flags,
   coordinated field forgery, invalid term/group bounds, module/sequence swaps,
   digest substitution and all four commit-byte values.  Until these pass,
   `analytical_cycle_annex_allowed` must be hard false.

### P1 — required for a citable calibration framework

1. Enforce the exact 3 sequences x 10 samples x 4 ordered layers population,
   including uniqueness of `(sequence, sample, module)` and nonzero diversity in
   group count and measured cycles per layer/sequence.
2. Report errors per layer and per sequence.  Validate the fixed slope and
   constants with leave-one-sequence-out or a frozen held-out sequence; do not
   fit and admit on the same 120 observations only.
3. Preserve the current global maximum-relative-error <=0.1% gate, and add
   maximum absolute error and per-stratum gates so a large layer cannot mask a
   small one through mean reporting.
4. Have a different-author hammer verify the repaired adapter, then authorize
   exactly one sealed calibration attempt.  A result hammer remains mandatory.

### P2 — robustness and presentation

1. Report residual distributions and slope sensitivity around 4 cycles/group;
   keep slope 4 frozen unless a new contract is independently reviewed.
2. Publish traffic as exact M1111DR2 diagnostic traffic by kind and layer, not
   as a new independent performance model.  Label cycle results analytical and
   calibration-only.
3. Bind the repaired source SHA directly in its contract and retain receipt-
   blind mutation tests as a permanent CI target.

## Sealed conclusion

M1281 has good fail-closed scaffolding, but its authority and transaction grain
are not yet trustworthy enough for downstream analysis.  The smallest safe
path is to repair the adapter boundary and commit projection, then repeat this
receipt-blind hammer.  No real calibration should be launched as a workaround.

`docs/359_DATE终局冻结_20260813.md` was not modified and remains SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

