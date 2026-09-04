# M1554 independent fail-closed hammer of M1553

Verdict: **NO-GO for the one D0/call-0 pilot**. The reviewed source is much narrower than M1549 and correctly removes the public `plane/module/call` seam, but two independent P0 witnesses remain. Neither witness scheduled request zero.

## What closed correctly

- The only public executable signature is `stream_actual_call(config)`.
- Replacing the module-level plane class name, selector name, or upstream root after import does not replace the ordinary captured objects.
- A subclass/custom plane cannot be supplied through the public signature.
- Symlinks and the tested lstat/open inode swap are rejected.
- Renaming and replacing a pathname after open does not change bytes read through the original descriptor.
- Product configuration, pilot/production release functions, and pilot/product/production CLI attempts remain blocked.
- The author test passes under CPython 3.6.8 and Python 3.10.18.

## P0 findings

1. **Mutable mapped inode after the one-time hash.** On both Python runtimes, a temporary exact-class plane is opened with the correct SHA; an in-place write through a second descriptor then changes `bit()` from 0 to 1 while `opened_sha256` still reports the admitted digest. The current fd identity check closes pathname replacement, but does not make the mmap bytes immutable for the scheduling interval.

2. **The captured selector still resolves mutable global `M`.** The closure captures the `selected_pilot_record` function object, not an immutable row. Replacing that function's module global with a delegating selector admits the already sealed D0 sample-11/call-4 payload after rewriting only the logical call/sample fields. It reaches scheduler construction as call 0; the hammer sentinel stops before request zero.

## Required successor boundary

Snapshot and pin the exact call-0 row, canonical relative path, and SHA within the closure during a clean import. Do not call a selector that resolves mutable module globals. Move verified compact bytes into immutable anonymous storage (the D0 bitpack is compact), or enforce an equivalently strong full-interval byte immutability guarantee. Re-run this independent hammer in a fresh process before any actual pilot.

No pilot, production population, product configuration, GPU, SSH, RTL, or EDA was run. `docs/359` remains at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
