# M2070 M2067 FC2 exact-continuation VCS source hammer (R2)

## Verdict

**PASS, 96/100, P0/P1/P2 = 0/0/2. One no-retry VCS execution is authorized.**

This review ran no VCS, EDA, license query, or GPU task. It inspected the frozen
R2 identity after M2068 rejected R1. `docs/359` is unchanged at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The parser `--static` check returned `PASS_M2067_STATIC_SOURCE_AND_FIXTURE`
with 960 workloads and 24 frozen sources. A relative filelist mutation fails
the absolute/cwd-independent identity gate. Extra runner arguments and missing
authority pins fail before the attempt latch, so they do not consume the
no-retry token.

## M2068 blockers

### P0-1 filelist / compile cwd — closed

All seven filelist entries are absolute regular files. The one-shot passes
`-f` with that absolute filelist while cwd is a fresh private WORK directory.
A reviewer mutation that replaces the wrapper entry with
`hw_autoresearch_nts07/rtl_m2067/...` is no longer absolute and would not
resolve under `WORK/`. The parser identity check requires the seven resolved
absolute paths exactly.

### P1-1 no-retry failure evidence — closed

The attempt directory is created and sealed before `lmstat`. The exception
handler publishes a no-replace sealed quarantine whenever the attempt latch
exists, including on license-preflight failure (WORK may be absent). The
quarantine records phase, command, license/compile/sim/completed-slot counts,
current slot, runner/parser/contract hashes, copied `.log/.json/.txt` evidence
without symlinks, optional raw `lmstat` bytes, and a raw-tree fingerprint of
attempt/work/result-stage. After a successful publish, a later exception cannot
emit a contradictory failure. Automatic retry is absent.

### P1-2 G192 negative alias attack — closed

The testbench issues two fail-closed attacks per workload, then resets:

- G96 chunk index 1 with `global_group_base=0` instead of 48;
- G192 chunk index 2 with `global_group_base=48` instead of 96.

Each attack requires both axes to reject, increment `alias_reject_count`, and
raise `protocol_error`. The parser requires
`alias_attacks=2`, `alias_attacks_g96=1`, `alias_attacks_g192=1`, and two
rejects per axis. Directed INT8 weights now depend on the translated global
weight-row index, so an output-tile row alias is visible to the integer oracle
as well as the address sideband.

## What remains sound

The wrapper is still an additive G48 continuation around frozen M2018/M803:
G96/G192 become 2/4 chunks, Acc24 is retained across inner resets, only the
final chunk commits externally, and
`weight_row_index = output_tile*source_group_count + global_group`.
The analytic bound `192*16*128 = 393216 < 2^23`. Logical cycles exclude
passive header wait and count accepted header plus reset/load/run. The fixture
is 960 unique sample-layer-role jobs (720 G96 / 240 G192), 40 samples, four
DSEC traces, eight FC2 layers, 1,843,200 integer checks per axis, 212,468
nonzero source codes and 0 negative codes.

## P2

1. The one-shot still publishes into namespaces whose directory names contain
   `vcs_r1_20260903` even though the schema is R2. The R1 VCS result namespace
   does not exist; this is naming residue, not a colliding artifact.
2. The parser is pinned by the runner authority hash and is excluded from the
   24-entry frozen source list to avoid a self-hash cycle. That split is
   intentional and must stay exact-pinned at launch.

## Authorization

Exactly one no-retry execution of

```
/opt/anaconda3/bin/python hw_autoresearch_nts07/dc_handoff/scripts/run_m2067_ep34_fc2_exact_continuation_vcs_one_shot.py
```

from the `SDformer` root, with the six `M2067_EXPECTED_*` pins below. Budget:
one `lmstat`, one compile, 960 serial `simv` slots. No automatic retry. A
successful output remains `PENDING_INDEPENDENT_RESULT_HAMMER_DO_NOT_CITE`
until a different-author result hammer. It is not full-FC wall time, energy,
or system speedup, and it does not promote the M2064 CPU ratio.

This review does not modify M2067 sources.
