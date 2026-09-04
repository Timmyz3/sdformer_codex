# M1560 — M1558 reduced-binary source independent review

## Verdict

Local format and synthetic consistency pass under CPython 3.10.18 and 3.6.8,
but remote integration wrapper authoring is **NO-GO** until one permit-gate P0
is fixed. Actual checkpoint load, GPU/SSH capture, release, retry, RTL and EDA
remain forbidden.

## What passed

- Exact M1458 inventory: 12 FC1 + 12 FC2 + 8 PATCH layers in canonical order.
- Exact populations: 44,640,000 FC tokens and 430,080,000 PATCH tokens consumed
  histogram-only.
- First-principles bounds: 7,528,535,874 raw bytes and 7,598,737,368 result
  upper bytes, below the strict 12-GiB runtime cap.
- Mixed-code and full producer synthetic roundtrips cover the fixed header,
  independently framed zlib, CRC/exact extent, canonical sample/layer/frame/token
  order, little-endian tail bits, uint16 nnz, sign, nonunit, zero-token retention,
  and incremental validation.
- PATCH emits only sample/layer/output-tile histogram/debt rows; no per-token
  PATCH payload is emitted.
- Diagnostic int8 coordinates remain explicitly non-authoritative for hardware
  quantization.

## P0 — permit issuer can be bypassed

`ReducedBinaryProducer` requires the exact `_PreloadPermit` type and the normal
issuer performs the intended fresh-path, estimate and strict post-result
16-GiB reserve checks. However, the closure that possesses the constructor
secret is retained as the module-global callable `_mint_permit`.

The independent regression directly called that object with zero free bytes.
It returned the exact producer-accepted permit type and consumed successfully,
recording `free_bytes_before=0` and `free_bytes_after_upper=-1`. Therefore the
type check does not prove that the checked issuer ran, and the preload permit
is not truly enforced.

The fix is narrow: remove the module-global raw mint capability and close permit
construction behind an issuer that always runs all disk/estimate gates. After
the fixed source passes an independent re-review, only remote integration
wrapper **authoring** may be authorized. Actual capture must still wait for a
separately sealed one-shot release and a later production-result hammer.

## Boundary

This review used only local source, fixtures and synthetic data. It did not
load a checkpoint, contact a remote host, use a GPU, execute capture/release,
or run RTL/EDA. It creates no AEE, cycle, traffic, energy, speedup or paper
claim.
