# M1321 ep34 decoder capture adapter — author receipt

Status: **source-only PASS; different-author source hammer required**.

M1321 prepares the narrow adapter needed after the final M1249 unified capture.
It recognizes exactly the decoder cohort at global sample IDs 10--39, requires
four D0--D3 calls per sample, and validates the captured `support_sign` payload
as two planes (`positive || negative`).  It does not pass that two-plane file to
the legacy M672 mapper as if it were a one-plane bitpack.

The FP32 payload is decompressed and checked word by word.  D0/D2/D3 admit only
exact `+0` and `+1`; D1 admits `+0` and one positive finite dynamic theta word,
with population-wide theta stability.  Positive support must equal raw nonzero
locations, the negative plane and padding bits must be zero, and the raw and
container SHA interfaces are checked.  A strict four-module weight identity
interface binds a future export to the selected checkpoint, but M1321 does not
load the checkpoint or export weights.

Eight synthetic tests pass under the pinned Python 3.10 interpreter.  The tests
cover malformed plane sizes, negative activity, near-one values, multiple D1
values, nonzero padding, zlib trailing data, raw-SHA drift, population/order
drift, bias drift and attempts to enter a production CLI mode.

No actual M1249 result was read because it does not yet exist at this authoring
cut.  There is no result-seal or result-hammer binding, no normalized bitpack
writer, no CPU replay and no cycle, traffic, speedup, energy, PPA or Table-A
claim.  A future additive successor must bind the actual capture result hammer,
then atomically materialize each positive plane and bind the final four weight
identities before reusing M1105/M785.
