# M1265a — M1258/R12 exact-TB reachability and claim-boundary audit

Date: 2026-08-30  
Mode: independent, read-only, source-only  
Decision: **PASS_READ_ONLY_REACHABILITY_AUDIT_NO_RELEASE_AUTHORIZATION**  
Score: **100/100**  
P0/P1/P2: **0/0/0**

## Frozen object

- TB: `hw_autoresearch_nts07/verif_m1258r12_c1_common_charge_protocol/tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12.sv`
- Exact SHA256: `e13d630f4cf2e2f7e0264dc2325218aee4cc580497be3b37deb1ff7a641ad302`
- The audit did not edit the TB, RTL, SVA, runner, workload, or `docs/359`.
- No VCS, EDA, GPU, remote, or release action was run or authorized.

## Findings

### 1. The child-output seam is executable from the sole initial block

There is exactly one `initial` block, at line 1301. Its unconditional phase sequence calls every top-level task that contains or reaches an M1258/R12 child-seam force/release:

| initial path | reachable child-seam operations |
|---|---|
| `directed_weight_first` | full child request force helper; full child release helper |
| `directed_psum_first_and_backpressure` | full child request force helper; child `issue_data_ready` override; full release |
| `directed_nonfirst` | full child request force helper; full release |
| `directed_ii2` | full child request force helper; child epoch/row/source retarget; full release |
| `reset_pending_cases` | full child force/release paths for all three reset cases |
| `sticky_fault_attacks` | full child force/release paths plus deliberate child valid/epoch attack overrides |
| `service_assumption_attacks` | no-ready child request helper and full release |
| `random_boundary_transaction`, called by the fixed 24-iteration loop | full child request force, child-ready backpressure/release, tuple-only retirement helper, final full release |
| every phase through `reset_dut` | defensive full child release before reset |

The exact source has 28 child force statements and 20 child release statements. Every one is inside a task on the above reachable call paths. There is no force or release on a parent `dut.issue_request_*`, `dut.core_issue_*`, or other parent connection. Thus the source statement `parent_connection_force=0` is statically supported.

### 2. Phase and PASS observability is reachable

The exact source contains 14 static `PHASE_M1258R12...` display sites and one exact `PASS_M1258R12...` display site. All 15 sites are directly in the sole initial block, after initialization and before the sole terminal `$finish`.

The 14 phase sites are the enter/complete pairs for directed, reset-pending, sticky attacks, service attacks, random outer phase, random transaction body, and integrated normal M935. The random-transaction pair is dynamically emitted once per iteration.

There is no `generate`, conditional-compilation guard, `if (0)`, `disable`, `return`, `break`, or `continue` that can statically hide the phase sequence. The sole `$finish` is line 1400, after the PASS display. Earlier exits are `$fatal` checks and therefore fail closed rather than manufacture PASS.

### 3. Random cardinality is exactly 24

The only `test_index` loop is lines 1347–1354:

`test_index = 0; test_index < 24; test_index = test_index + 1`

`test_index` is not assigned elsewhere in the body; there is no loop escape. Therefore `random_boundary_transaction(test_index)` is reached exactly 24 times on a non-failing execution, and both transaction phase displays are reached exactly 24 times.

### 4. The integrated normal M935 evidence is real and single-shot

There is exactly one executable `normal_m935_completion();` call, directly between the integrated enter/complete displays in the sole initial block. The task performs:

1. clean reset and legal-mask/prep admission;
2. `load_normal_task(16'h9001)` through the public prep interface;
3. `serve_normal_beat(1'b1, 0)` and `serve_normal_beat(1'b0, 1)` through the external request/response services;
4. exact checks for two issue accepts, one row completion, one task completion, epoch `16'h9001`, and no protocol error.

The combined three-task region (`load_normal_task`, `serve_normal_beat`, `normal_m935_completion`) is byte-identical to frozen R11. Individual task hashes match R11:

- `load_normal_task`: `fa86553341c84a31e0715dd751ce6f41161eed7557932bb35fbbaafc20b9a669`
- `serve_normal_beat`: `9c27568bd89e590de7c40fad88c1eeedb818875ab7ac8f01c492965abff838c5`
- `normal_m935_completion`: `bf30589e69f52f856edb269fcdde20f837eb1a646264d22e60d0ca70ef6a51f4`
- combined region: `58142191345b94e824fb7d29eaa04a693e5ab8a4ab020211aa6bdfa8f43fe5f6`

`reset_dut` first releases every child seam field. None of these three normal tasks forces any child output. Consequently the normal phase is integrated frozen-M935 evidence, while directed/reset/attack/random evidence remains boundary-only.

### 5. Claim boundary is internally consistent

The exact PASS token correctly says:

- `boundary_only=true`
- `integrated_random=false`
- `parent_connection_force=0`
- `child_core_output_seam_force=1`
- `integrated_m935_claim=false`
- `integrated_normal_m935_evidence=true`
- all timing, cycle, speedup, PPA, energy, system-speedup, and headline claims are false.

The source topology supports those boundaries. This audit establishes static reachability and exact-byte provenance only. It does not prove that a future simulation reaches PASS, that SVA failures are zero, or that timing/PPA/performance claims exist.

## Disposition

The exact R12 TB is suitable as an input to a separately hammered exact-byte release review. This M1265a audit itself does **not** authorize release or execution.
