# M1223 — M1208 40-sample capture failure read-only forensic

## Verdict

M1208 completed all 40 forward passes and left substantial partial data, but
it failed at the exact per-module runtime-call gate.  The canonical result was
never published, the attempt is consumed and non-retryable, and the fixed
staging is not recursively sealed.  It must remain failed evidence.

The independent read-only audit verifies two useful subsets: all 480 attention
NPZ form an exact 40×12 sample/block matrix and pass manifest SHA plus required
Q/K/gate-array checks; all 640 retained payload files form 320 complete
raw/support pairs whose filename hashes reconstruct exactly 40 calls for each
of four C1 and four decoder modules.

The full per-record 259-module actual call distribution is **not recoverable**.  It lived
only in `UnifiedHookWriter.records`; the source checks equality in `close()`
before writing ordered records or any runtime manifest.  The process has
exited, so no file can reconstruct its ordered rows or observed dictionary.
M1224's source/topology correction does identify the aggregate failing set:
the twelve hook-dead modules are the twelve `.sn_v` ATLIF modules, **not** the
twelve attention parents.  That correction explains 9,880 actual calls versus
10,360 expected (480 missing), but it does not recreate the lost rows.
M1223 binds that correction to the sealed
`reviews/m1224_m1208_capture_contract_first_principles_audit_r1_20260830`
authority: review JSON `56372da5...`, manifest `677bb081...`, and outer seal
`9a858b44...`.

## Exact failure mechanism

The frozen expected inventory has 259 modules:

| Category | Expected modules | Expected calls | Actual recoverability |
|---|---:|---:|---|
| C1 Conv3x3 | 4 | 160 | filename-exact: each 40 |
| Decoder ConvTranspose | 4 | 160 | filename-exact: each 40 |
| ATLIF | 105 | 4,200 | M1224: 93×40 observed, 12 `sn_v`×0; rows lost |
| FC1 | 12 | 480 | lost with in-memory records |
| FC2 | 12 | 480 | lost with in-memory records |
| Patch embed | 8 | 320 | lost with in-memory records |
| BatchNorm | 78 | 3,120 | lost with in-memory records |
| Q/K projections | 24 | 960 | lost with in-memory records |
| Unified-writer attention | 12 | 480 | M1224: not the deficit; separate writer complete |
| **Total** | **259** | **10,360** | full actual map unavailable |

`StrictWriter.close()` initializes all expected names to zero, counts
`self.records`, and requires the resulting nested dictionary to equal 40 for
every name.  It raises only the aggregate message
`per-module runtime call coverage is not exactly 40`; it does not persist the
dictionary or its delta.

The R1 run sequence calls `writer.close()` before `profiler.close()`,
`unified_ordered_records.jsonl`, execution/operator/ATLIF summaries, and the
result manifest are written.  The exception path adds only `FAILED.json`.
Consequently those later artifacts do not exist by construction.

Payload files do not rescue the missing map.  `_payload()` explicitly retains
files only for `c1_conv3x3` and `decoder_convtranspose`; all other hook rows are
ordered-statistics-only memory objects.  The retained global-order positions
normalize to `[220,223,226,229,232,236,240,244]` for every sample with stride
247.  This is exactly the 247 live-hook pattern used by M1224 to identify the
twelve `.sn_v` ATLIF hooks as dead.  The staging filenames alone do not carry
those names, so M1224's source/topology proof is required for attribution; no
ordered row should be synthesized from this inference.

## What is strictly verified

The fixed staging is mode 0700, contains 1,122 regular files and no symlinks,
and was stable across two full read-only hash passes.  Its canonical
path/size/SHA inventory digest is
`6bd6ce19e38f9611fafcf8c1d91d304b13d93fb1a2887006f0eb27dc69058a7b`.
This is a local forensic snapshot binding, not a substitute for a remote
producer seal.

Attention evidence is the strongest salvageable subset:

- manifest SHA `edbe96ce...`, 480 records and 480 NPZ;
- each sample 0–39 has all 12 blocks exactly once;
- each block appears 40 times;
- every recorded SHA matches its NPZ;
- every NPZ opens with `allow_pickle=False` and contains nonempty
  `q_bits_packed`, `k_bits_packed`, and `gate_q17`;
- NPZ inventory digest `e4513b08...`;
- run context binds ep29 checkpoint `2144dfd6...`, config `c7b5b994...`, exact
  checkpoint load, BN policy `no_running`, and topology 105 ATLIF/12 attention.

This admits “independently verified partial attention capture at the observed
failed staging snapshot,” not “M1208 capture complete” and not a paper result.

For the 1.083 GB retained payload directory, filenames strictly prove 320
complete two-file pairs and the eight module/sample call populations.  Because
the per-record metadata and embedded compressed/support SHA fields were in the
lost ordered rows, payload numerical semantics, shapes, source sample binding,
and raw-tensor SHA cannot be fully revalidated from filenames alone.  Those
files are forensic partial payloads, not admitted tensors.

## Unrecoverable state

The following existed only after or inside the failed process and cannot be
recovered from this staging:

- `StrictWriter.records` for all nonretained categories;
- the expected-versus-observed per-module delta that triggered the exception;
- unified ordered JSONL and its per-record payload metadata;
- execution trace, operator runtime, ATLIF activity and final module inventory;
- canonical manifest, RUN_COMPLETE and recursive producer seals.

The 247 stride must not be used to blame attention parents: M1224 corrects that
interpretation to twelve `.sn_v` ATLIF hooks.  Even with that correction, the
missing in-memory rows and observed dictionary remain unrecoverable.

## Minimum recovery and successor plan

1. Under a new, separately hammered authority, copy the fixed staging to a
   disjoint immutable **failed-forensic** package, generate a complete recursive
   manifest, verify it, and double-seal it.  Never rename it to the M1208
   canonical success path and never add RUN_COMPLETE.
2. Author a new capture source that streams every unified hook record per
   sample (or checkpoints it atomically), and always writes
   `runtime_call_coverage.json` containing expected, observed and per-name delta
   **before** enforcing exact40.  Its exception path must seal diagnostics.
3. Replace the invalid universal "module `forward` hook must fire" premise for
   `.sn_v` with an execution-point appropriate hook/counter, or explicitly bind
   those twelve modules to the specialized attention path.  Do not merely
   delete them from coverage.  Preserve exact40 for every live execution point,
   and keep the specialized attention 40×12 gate mandatory.
4. Because M1208 is consumed and the complete unified records were never
   persisted, a valid unified result requires a fresh namespace and a new
   authorized 40-sample inference.  Reusing the partial attention subset may be
   considered only as an explicitly separate artifact with exact identity
   binding; it cannot turn M1208 into PASS.
5. Before any expensive successor, run a separately authorized diagnostic
   sample with the new persistent coverage map.  Use its exact delta to repair
   the inventory/hook boundary, then hammer the final source and launch once.

M1223 performed no remote write, rerun, GPU compute, or EDA.  It authorizes
neither salvage mutation nor a successor launch; those require new contracts
and independent hammers.  `docs/359` remains `dedde7ce...`.
