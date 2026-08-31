# M218 tagged slice-service premodel

This frozen-H67 premodel is the go/no-go screen for a tagged, multi-outstanding,
16-lane-sliced FC2 weight service.  It checks all 120 payload files and counts
six 128-bit active-bank reads per source/output-block update, finite response
latency, outstanding capacity, memory initiation interval, same-context
hazards, sliced result drain and token completion.

The executable M216 window order produces 73,380,812 K8 group commands.  The
older 70,657,362 value is retained only as an independent-bank-queue oracle;
it is 3.854446% lower because it may reorder across the token instead of
respecting M216 windows.  K1 and K8 both perform exactly 2,477,402,364 active
bank-slice reads and transfer 39,638,437,824 weight bytes.

At the primary `L4/O8/II1` point, service cycles are 2,552,566,588 for K1 and
515,449,096 for K8, a 4.952122x service speedup.  K8 retains 97.837193% of the
`L1/O8/II1` oracle throughput.  Without pretending to know exact temporal
overlap between the admitted M216 frontend and this service, interval
composition gives a conservative 4.214619x speedup lower bound.

This is a fixed-latency, in-order, pre-RTL model.  It admits opening M218 RTL;
it is not VCS calibration, out-of-order-response proof, complete FC2/FFN,
physical PPA, system speedup, or a paper headline.
