# M218 admission index

M218 is admitted as a standalone tagged, slice-coalesced sparse-FC2 service
island.  The index binds the frozen-H67 service premodel, its independent
91/100 review, exact Synopsys VCS, the independent 92/100 RTL review, exact
3 ns logic-only DC and the independent 89/100 DC review.  Every review has
P0=0.

At the frozen `L4/O8/II1` point, K1 service takes 2,552,566,588 cycles and K8
takes 515,449,096 cycles: 4.952122x.  Conservative interval composition with
the standalone M216 frontend gives a 4.214619x lower bound.  Both sides read
exactly 2,477,402,364 active bank slices and 39,638,437,824 weight bytes.  The
mechanism is therefore context/control amortization, not hidden weight-work
elision: K8 reduces six-slice context-update requests by 82.227963%.

The executable RTL passes numeric, conservation, out-of-order response,
backpressure, slot/context reuse, identity, duplicate, flush and timeout tests.
The 3 ns pre-macro DC point is 88,851.042296 um2 with +0.6872 ns setup slack
and 0.0000 ns hold slack.  It contains no SRAM macro.  Its 18,432 context FF
bits are 87.289% of sequential cells, and hold repair adds approximately
7.820% area, so this area is intentionally not treated as physical PPA.

Native-cropped K1 area/energy sensitivity, a context/weight macro adapter,
Formality, SAIF/PTPX, connected frontend-service cycle validation, complete
FC2/FFN and system/headline claims remain outside admission.
