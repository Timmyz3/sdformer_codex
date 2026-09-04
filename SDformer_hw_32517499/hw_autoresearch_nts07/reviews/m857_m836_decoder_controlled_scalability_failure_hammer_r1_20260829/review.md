# M857｜M836 decoder controlled-scalability failure hammer

## Verdict

**PASS100 failure audit; the sole M836 production identity is permanently consumed and all decoder cycles/speedups remain noncitable.** The exact authorized run reached `RUN_PRODUCTION_DRIVER`, entered the first `D0 / A1_OSG / t0` address-timed schedule, and was then externally terminated in a controlled way after the frozen M768 post-schedule implementation exposed an intractable cycle-by-request scan. The fail-closed trap published a valid double-sealed failure receipt, moved the empty private stage to an empty partial-artifact directory, left no canonical result and left no orphan process.

Target defects: **P0=1, P1=0, P2=0**.

## Sealed terminal state

- Failure directory: `results/m836_m785_h67_decoder_physical_residency_cycles_r1_20260829.failed_or_incomplete.1114302.19219.32716`
- Failure receipt SHA256: `f9134f684db264fd504941261866568140a944b36dec8d9294d0c6ca8a70e3fe`
- Failure manifest SHA256: `6f30cb91f17789b8197a2478bd1bccfe6745995cb255ab451139b9bcaaa03432`
- Failure outer-seal-file SHA256: `ea2fe94623e85aeb5175fbe16bdcc81ec1fda5fdf6adf94286640f337be0f988`
- Driver log SHA256: `9391796e084b23eaa96e62ae66263062a2d413344f1190543777f3eed7fb7854`
- Consumed attempt: `results/.m836_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed`
- Attempt receipt SHA256: `ab2fe3f1a6a2f2b43a77656fc4c4b70ebff61d2b8f4c61d9d828cba9ff8f5dd8`
- Attempt manifest SHA256: `608e77024dc411c19e71514f2f5c853a0cb9a6764661c09a8abddd43609b7689`
- Attempt outer-seal-file SHA256: `8b93d2668869b099af8aac064eefbd1295fa1687b7cea4f699cef1bacbaf262e`

Both populations and both outer seals recompute. `failure.json` records return code `143`, phase `RUN_PRODUCTION_DRIVER`, status `FAILED_OR_INCOMPLETE__NO_CYCLES_CITABLE`, and five explicit false claim flags. The canonical result is absent. The partial-artifact path is a regular empty directory. The only related top-level objects are the consumed attempt, the sealed failure directory and its empty partial artifact. No M836/M785/M768 production process remains.

## Independent failure localization

The stdout portion of `driver.log` proves the attempt-required preflight completed with zero schedule rows. The stderr traceback then terminates inside the first production call at frozen M768 lines 566--567, specifically the per-cycle construction of `reasons` from `waiting` rows. It is not an identity, input, resource, address, dependency, arithmetic or assertion rejection.

The frozen call chain is mechanically exact:

1. M809 loops two populations, three configurations, every record and ten timesteps. Its completed-result assertion requires `(40 + 120) * 3 * 10 = 4,800` detailed schedule rows.
2. The first configuration is `A1_OSG`; normalized M686 ordering makes the first record `zurich_city_09_a / sample 0 / module 0`, hence the interrupted invocation is `D0 / A1_OSG / t0`.
3. M785 line 143 first materializes all expanded requests with `rows = list(requests)` and then calls frozen M768.
4. M768 first materializes every `ScheduledRequest`; after that, lines 559--579 iterate every cycle and scan the whole scheduled list twice to reconstruct waiting and inflight sets. This postpass is `O(C * R)` in total cycles `C` and expanded requests `R`, in addition to `O(R)` materialization.
5. M854's bounded first-row diagnostic reported `9,582,057` compressed transactions and `38,672,612` expanded requests for this exact first invocation. This audit independently confirms the row identity, loop population and `O(C * R)` implementation path. Those two cardinalities are retained only as **diagnostic scalability evidence**; they are not cycles, speedup or a completed production row.

The private stage was born at `03:28:48` and the failure receipt at `04:00:03`, so the run spent roughly 31 minutes in the first of 4,800 intended schedule invocations without returning even one detailed row. Return code 143, the terminal `KeyboardInterrupt`, the correctly executed fail-closed trap, the empty quarantined stage and the absence of orphans jointly classify this as a controlled external termination of a non-scalable implementation, not a completed model result and not a silent crash.

## P0 finding

**M857-P0-1 — the frozen production implementation is not executable at its admitted population scale.** Even the first row expands to 38.67 million requests, while the postpass scans the entire expanded population for each cycle. The release had 4,800 such invocations and no row checkpoint or aggregate-only path. Therefore this M836 identity cannot produce a decoder-complete result in bounded practical time, and the consumed identity may not be retried.

This finding invalidates no frozen transaction semantics. It invalidates only the scalability of the current production executor and every claim that depends on a completed M836 result.

## Additive successor and event-sweep miter gate

Only a new additive identity with new attempt/result/failure paths may proceed. M768, M777, M785, the payloads, resource tuple, three configurations, dense commit sequence, D1 exclusion and legal K8-vs-equal-service-K1x8 denominator remain exact-SHA parents.

Before another true release, the successor must pass all of the following:

1. **Streaming construction:** remove production use of `rows = list(requests)`, the full `scheduled` return payload, and per-cycle list scans. Production must retain aggregate counts, ordered address/commit hashes, exact total cycles and cycle-class totals without serializing expanded request rows.
2. **Exact event sweep:** represent issue points plus waiting, dependency and inflight half-open intervals with endpoint events. At every integer span, preserve the frozen precedence exactly: active issue, then dependency/inflight, weight-bank, psum-bank, memory, and compute.
3. **Old-vs-new exact miter:** on the complete frozen synthetic suite and deterministic bounded real prefixes, require equality of request endpoints, produced-token readiness, total cycles, all cycle classes, compressed/expanded counts, transaction-address SHA256 and commit-sequence SHA256. Tests must cover 1RW/1R1W conflicts, outstanding limits, same-cycle response-slot reuse, simultaneous issues and every wait-reason precedence edge.
4. **Full first-row scalability gate:** replay the exact M854 first-row identity under the successor; require the same `9,582,057 / 38,672,612` cardinalities, bounded peak memory, completion of aggregate scheduling, and no result publication. The full-row output remains diagnostic until a new release.
5. **Population projection gate:** sample every module, configuration and population class, then demonstrate a bounded wall-time/RSS projection for all 4,800 rows. If the projection does not fit the declared production window, shard by exact row identity with double-sealed, no-clobber partials and an exact aggregate miter before seeking release.
6. **Fresh authorization chain:** source-only candidate, independent source hammer, separate true release and fresh final-launch hammer. Only one new production replay is permitted. A completed canonical result still requires a fresh independent result hammer before any cycle or ratio is citable.

Not authorized: rerunning M836; deleting, moving or rewriting its attempt sentinel; reusing its release; starting successor production; Table-A insertion; decoder/full-network/system claims; VCS, EDA, license, GPU, remote or training work under this review.

`docs/359_DATE终局冻结_20260813.md` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
