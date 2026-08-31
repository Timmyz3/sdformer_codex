# M942 — decoder D1 exactness and four-layer slice first-principles audit

## Verdict

**Audit PASS 100/100, P0/P1/P2 = 0/0/0.**  D1 does not lack a payload.
Its 10 captured inputs are exactly representable as a one-bit mask plus one
immutable FP32 scalar, and this review independently reconstructed all
92,400,000 FP32 input elements with **10/10 byte hashes equal**.  What failed
is the proposed accelerated numerical route: pre-folding the scalar into FP32
weights is not bit-exact.

Therefore the DATE-window-safe path is:

- keep D0/D2/D3 as the exact-binary decoder support subset;
- charge D1 through the same common-resource ledger as a dense, nonheadline
  diagnostic;
- do not call this partial population decoder-complete.

No simulator, EDA, GPU, remote host, or network operation was started.  The
only computation was a read-only NumPy unpack-and-hash reconstruction of the
ten sealed D1 bitpacks.  Existing evidence and `docs/359` were not modified.

## Root cause by layer of the stack

| Suspected cause | Finding |
|---|---|
| Missing D1 activation or original FP32 weight | False. M686 contains ten D1 bitpacks, the scalar identity, original/folded weights and ten output hashes. An admitted INT8/fixed-point decoder bridge is still missing. |
| Shape or model hook error | False. D1 is consistently `[10,1,770,30,40]`, consumes a `[770,192,3,3]` bias-free ConvTranspose2d, and its hook order/threshold identity is sealed. |
| Quantization/numerical non-equivalence | **True.** D1 emits `{0, theta}`, not `{0,1}`; moving theta into FP32 weights changes rounding and accumulation bits. |
| M809/M836 runner failure | Real but independent. It prevents a canonical production result; it does not cause the D1 miter mismatch. |
| M896 limitation | Real but orthogonal. Its real selector is hard-coded to record 0, D0/sample0/A1/t0. It never exercised or repaired D1. |

The scalar is `theta = 0.9999954104423523` (`b3ff7f3f`). Across ten samples,
17,085,826 of 92,400,000 inputs equal theta and the remainder are exact zero.
Reconstructing each input as `where(mask, theta, +0.0)` reproduces every sealed
raw-input SHA.

In contrast, `conv_transpose(mask, float32(theta*W))` differs at 67,924,171 of
92,160,000 output elements (**73.7024%**) across S10. Per-sample maximum
absolute error is `4.005e-4` to `4.158e-4`. The high bit-mismatch fraction is
not evidence of a large task error; it is decisive evidence that the route is
not FP32 bit-exact. Real-number distributivity cannot be substituted for an
FP32 miter.

The other candidate, `theta * conv_transpose(mask,W)`, is equivalent over real
arithmetic because bias is null, but moves the rounding point after
accumulation. It remains unmetered and unadmitted.

## What each milestone actually says

- **M624** is a historical availability audit from before the decoder capture:
  it saw zero ConvTranspose rows and missing bitpacks/deployment weights. M686
  later supplies activation payloads and original FP32 weights, but not the
  missing admitted INT8/fixed-point bridge or full co-executable schema. M624
  is not evidence that D1 currently lacks a captured input.
- **M686/M692** admit 30 exact `{0,1}` D0/D2/D3 records and ten exact `{0,theta}`
  D1 input representations. They explicitly do not admit D1 numerical
  deployment equivalence.
- **M700/M739** report an official-Prosperity product-vs-bit opportunity of
  3.087586x for D0/D2/D3. D1's 2.615818x is a separate diagnostic. Both are
  external phase-sum support cycles, not our full decoder latency.
- **M785** accepts D1 structurally, but replaces all three candidate configs by
  the same full-shape dense-FP32 billing stream. This is conservative cost
  retention, not an executable Acc24/INT8 numeric implementation.
- **M809→M836** preserves that policy and excludes D1 from headline totals.
  The final attempt ended with return code 143, so it produced no citable
  canonical cycles.
- **M896/M925/M939** prove scalable scheduler equivalence only for the first D0
  row. Scheduler compression does not solve the D1 arithmetic contract.

## Lowest-cost representative slice

Reuse the recursively sealed M925 D0 row rather than rerunning its 15.6-minute,
10.2-GiB diagnostic. Under a fresh identity, add sample0/A1-OSG/t0 rows for:

1. D1 as `COMMON_CHARGED_FULL_SHAPE_DIAGNOSTIC_NONHEADLINE`;
2. D2 as `EXACT_BINARY_SUPPORT`;
3. D3 as `EXACT_BINARY_SUPPORT`.

All rows must bind the frozen M785 resource: 96 lanes, 245,760 B SRAM, Acc24,
3 ns, and 192 B/cycle external service. Each needs expanded/compressed counts,
the six cycle classes, address and commit hashes, elapsed/RSS, and an explicit
numeric-route/headline flag.

This produces **three exact support rows plus one charged diagnostic**, not
four exact accelerated layers. It is legal to report a D0/D2/D3 exact-binary
subset and a separately billed D1 sensitivity. It is not legal to write
"decoder complete", "full decoder speedup", or "D1 lossless sparsity".

## Runtime budget and STOP gates

M925's D0 anchor is 38,672,612 expanded requests, 937.46 s and 10,716,244 KiB
peak RSS. D1's frozen dense billing has a closed-form 16,688,570 requests,
giving a rough 6.7-minute linear proxy. Using official support activity only as
a first-order sizing proxy, D2 is about 3.93x D0 (61 minutes, 40 GiB) and D3
about 13.03x D0 (204 minutes, 133 GiB). These are sizing estimates, not launch
promises or cycle results.

Before a full row, a fresh selector must run only 1K/10K/100K prefixes for
D1/D2/D3 and project mapper state, scheduler state, elapsed and RSS. Stop if
the two-times memory margin exceeds available memory/commit headroom, or if a
fresh per-layer timeout cannot be justified. The D0-specific 2,715 s timeout
must not be copied to D2/D3.

M925 counted only 532 MB of scheduler state but reached 10.7 GiB process RSS;
the large gap points upstream to materialized contributor/transaction objects.
If D3 fails the prefix gate, the correct repair is to stream contributors by
destination/stripe before M896, then exact-miter prefixes against M785. It is
not to launch the current materializing path with a larger timeout.

## Can D1 be repaired exactly?

At the payload boundary, **yes**: mask + exact FP32 theta reconstructs the
original input byte-for-byte. As a currently admitted accelerated compute
route, **no**.

The clean repair is a scale-aware source descriptor that carries the mask and
one layer-static theta while retaining the original weight. It still needs a
bit-accurate numeric engine, a 10/10 output miter, and fair area/throughput
accounting. Existing Acc24/INT8 hardware cannot be assumed to execute the
frozen FP32 contract. If that repair requires a separate FP32 datapath, stop
and retain the dense diagnostic.

The alternative is a new deployment checkpoint whose D1 emission amplitude is
exactly one or a power of two, followed by PAFT/fine-tuning, valid-set AEE and
complete payload/cycle reruns. That can create a clean exact integer route, but
it changes checkpoint identity and is not the minimum DATE-window action.
