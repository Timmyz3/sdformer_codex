# M1276 decoder/Table-A readiness read-only audit (2026-08-30)

## Verdict

**LET_THE_SERIAL_RUN_FINISH; DO NOT RETRY IT.**  At the observation cut the sole
M1111DR2 process (PID 4122290) is healthy and has emitted 73/120 canonical-order
call rows.  The attempt is already consumed, the lock is held, the canonical
result is absent, and no action should be taken against the producer.

Finishing row 120 can directly close only an **H67-ep35 decoder-only,
D0--D3-complete, common-resource, address-timed diagnostic cycle/traffic
result**.  It cannot directly create a full-network Table-A row: the runner has
one `M1105DR2_EXACT_TYPED_K8` configuration, deliberately emits no ratio, has
no energy/PPA/accuracy receipt, and sets `final_checkpoint_rebind_required=true`.
The M653/M698 registry code therefore must continue to reject it as a
full-system production bundle.

This audit is read-only with respect to the live run.  It launched no replay,
EDA, GPU, remote command, or production job and did not modify `docs/359`.

## Observed live state

| item | observation |
|---|---|
| process | PID 4122290, runner SHA `1167258c...9d746`, approximately one CPU core |
| progress | 73 JSONL call rows; ordinals 0--72 are present |
| population order | 30 samples x D0,D1,D2,D3; 120 rows required |
| attempt | `.m1111dr2_m1105dr2_decoder_only_production_attempt_consumed` present |
| lock | `.m1111dr2_m1105dr2_decoder_only_production.lock` present |
| canonical result | absent while producer is live, as required |
| retry | forbidden; maximum attempts is one |

The row stream is not a safe completion token.  In particular, `wc -l == 120`
may be visible before the result JSON, completion token, atomic seal, no-replace
rename, and lock release finish.

## Exact success/failure detection after PID exit

A successful completion requires all of the following, conjunctively:

1. PID 4122290 is gone and the canonical lock directory is gone.
2. `results/m1111dr2_m1105dr2_decoder_only_address_timed_production_r2_20260830`
   exists as a non-symlink directory; no matching `failed_or_incomplete.*.quarantine`
   is treated as success.
3. Its payload set, excluding `.m1111dr2_atomic_seal`, is exactly
   `m1111dr2_decoder_result.json`, `m1111dr2_decoder_call_schedule.jsonl`, and
   `RUN_COMPLETE.txt`.
4. The JSONL has exactly 120 canonical rows, ordinals 0--119, three sequences x
   ten samples x D0--D3, continuous transaction ordinals and cycle intervals.
5. `RUN_COMPLETE.txt` is exactly
   `M1111DR2_DECODER_DIAGNOSTIC_COMPLETE__RESULT_HAMMER_REQUIRED` plus newline.
6. `run_m1111dr2_m1105dr2_decoder_only_production_zero_arg.py`'s existing
   `validate_publish_candidate(result_dir)` passes, including recomputed SHA,
   traffic conservation, D1 theta word 1065353139, common resource, and claim
   projection.
7. `.m1111dr2_atomic_seal/SHA256SUMS` covers the exact three payload members and
   its outer seal verifies.

If the PID exits without all seven conditions, the run is failed/incomplete.
Because the consumed-attempt marker persists, it must be forensically reviewed,
not restarted or reconstructed from the partial JSONL.

## What row 120 makes directly reusable

| artifact or code | usable immediately after a successful result hammer | boundary |
|---|---|---|
| M1111DR2 120-row JSONL | per-call and aggregate diagnostic cycles, six transaction-kind counts/bytes, bank/port stall classes, exact address/dependency/schedule digests | ep35 decoder-only, typed-K8 only; not a speedup |
| M1105DR2 source | D0--D3 canonical order, 96 lanes, Acc24, 240 KiB, 192 B/cycle, bank/latency/commit rules | frozen ep35 payload identity |
| D1 current bridge | exact input reconstruction for `{0, theta}` with theta FP32 word 1065353139, no coercion to one, no weight folding | proves activation encoding identity, not a final-checkpoint/end-to-end decoder miter |
| existing runner validator | strict read-only validation of the published three-file result and seal | should be reused by a different-author result hammer; no schedule regeneration |
| M653/M698 registries | fail-closed schema and native Synopsys/macro evidence checks | cannot ingest this decoder-only one-row diagnostic as Table-A production |

The published ep35 result is valuable as a methodology/debug anchor and as a
decoder share/traffic characterization.  It must not be joined with final
checkpoint C1/FC/attention activity to fabricate a mixed-checkpoint system row.

## What still blocks decoder-complete memory-inclusive Table A

### Final-checkpoint binding

The final selected checkpoint invalidates the ep35 decoder payload population
and all cycles/traffic derived from it.  M1249, once legally launched and
hammered, supplies the right raw material without another checkpoint load:
40 samples, exact FP32 plus support/sign payloads for all four C1 and four
decoder operators, and ordered/operator/ATLIF/attention records.  Decoder rows
are the 30-sample decoder cohort (sample IDs 10--39), four modules per sample.

However, M1249 is a capture, not an address-timed replay.  A fresh final-bound
decoder authority must derive and seal:

- the final checkpoint/config/profile and four decoder weight/bias identities;
- the 120 decoder call order and exact FP32/support payload SHAs;
- final D1 value-class identity and numerical bridge;
- a fresh transaction population, cycles, traffic, and energy binding.

### D1 bridge

The ep35 theta constant cannot be copied to the final checkpoint.  For the
final 30 D1 calls, the bridge must determine from the captured FP32 words
whether each tensor is exactly `{+0, theta_final}` with one stable theta word.
It must reconstruct every raw FP32 tensor from `(support bit, theta_final)` with
zero SHA mismatch.  It must then preserve a typed scaled-binary source through
the final checkpoint's original ConvTranspose weights/bias and prove zero
mismatch against the reference output (or retain the typed multiply).  Silent
coercion to one and unproved theta-to-weight folding remain forbidden.  A
multi-valued/nonuniform D1 must fail this fast path and use a typed FP32/fixed
descriptor route.

### Full-network Table-A join

M1111DR2 is decoder-only.  A paper Table-A row additionally requires one
coherent final-checkpoint full-network invocation with all operator adapters,
the decoder inserted in global order, common numerator and matched baseline
ladder, and memory/physical evidence.  In the executable registry this means,
at minimum:

- a decoder-complete trace manifest whose operator scope includes
  `ConvTranspose2d`, rather than the old M51 trace with 150 missing records;
- distinct executable rows on the same population/resource (not just typed-K8),
  with the fixed M527-style throughput numerator and excluded work still
  charged;
- logic + SRAM + DRAM energy per inference, 17-macro area/power crosschecks,
  STA/PT/SAIF/PTPX provenance, and accuracy identity;
- at least three DSEC sequences with the declared density strata and a fresh
  independent final hammer.

Consequently, **no existing single script directly turns row 120 into a
decoder-complete, memory-inclusive full-system Table-A row**.  The existing
M653/M698 builders are validators for a future evidence bundle, not generators
of the missing replay, power, macro, or accuracy evidence.

## Minimal no-rerun post-processing sequence

1. Wait for the serial producer to exit naturally.  Do not poll by opening or
   rewriting its work file, stop it, seed a successor, or retry it.
2. Apply the seven-condition completion test above.  On failure, seal a
   forensic STOP; do not promote the prefix.
3. Author one different-author **result hammer** that calls the existing strict
   validator on the already published directory and independently recomputes
   the 120-row order, aggregate cycles/traffic, SHA chain, exact D1 rows, and
   atomic seal.  This is the only required post-processing of the expensive
   ep35 run; it performs no replay.
4. Keep the hammered ep35 output in a decoder-only diagnostic annex.  Do not
   write an ep35 Table-A production row and do not generate dense/bit ratios by
   algebra from official Prosperity or other component data.
5. After final selection and M1249 capture/result hammer, build a fresh
   checkpoint-bound M1105/M1111 successor by reusing the scheduler code but
   binding the new payload/weights/D1 miter.  This necessarily performs one CPU
   replay of the final payload; it does not rerun the ep35 producer and needs no
   new GPU capture.
6. Join that final decoder replay with the final full-network operator adapters
   and matched baseline rows.  Only then run/attach SAIF/PTPX, SRAM/DRAM energy,
   macro/STA evidence, construct the M698-style production bundle, and hammer
   Table A.

This ordering avoids repeating the current long computation while preserving
checkpoint and comparison integrity.  The only unavoidable recomputation is
the final-checkpoint CPU schedule, because activity-dependent cycles and traffic
cannot be rebound by changing a SHA field.

## Frozen identities

- M1111DR2 runner SHA256: `1167258c228631b73ca1784ae57db19e8f0fbe709efa34f369585c508bc9d746`
- M1105DR2 source SHA256: `b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4`
- M1111DR2 contract SHA256: `821819b00503b91a8fb8dfca8fe000208e10746e751a3815131dc8ff1cbed515`
- M653 registry validator SHA256: `97ce23afec30f91acfc612c06d4d5344680922842a04fd0c747675899156b9fd`
- M698 registry builder SHA256: `81fdc6e28e3940652f9afa65780d7539fde91d26fdcb6bef49cef9f6a260849e`
- M1249 release source SHA256: `5fbcc4d287f3ffd3b1c9994efa24245e5e3828927cdac925c1a35d8a88a19219`
- M1227 capture substrate SHA256: `11826d81c257bb0a14def4ab620be6c3971e4eea4175d6701e88de055140116b`
- protected `docs/359` SHA256: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

