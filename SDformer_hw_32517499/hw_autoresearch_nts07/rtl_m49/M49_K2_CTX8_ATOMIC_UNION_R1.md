# M49 K2-CTX8 Atomic Union-Source Engine

## Hardware contribution

M49 turns the M45 dual-destination transaction rule into a standalone
P8-L96 RTL engine.  A scheduler allocates up to eight signed-19x96 destination
contexts and explicitly launches either one context (K1) or a frozen pair
(K2).  In every active cycle, each of eight banks selects the lowest remaining
row in the union of the two destination masks.  That physical weight row is
read once and independently adds to, subtracts from, or bypasses each
destination accumulator.  Two destinations that require different rows in the
same bank necessarily consume different cycles.

The response metadata FIFO has 16 real enqueue/dequeue entries.  The complete
FIFO stores 16 full 96-lane signed-19 vectors plus tag and source count.  A
final K2 response is accepted only with two completion credits, where one
credit may come from a legal same-cycle output retirement.  Both vectors are
written at `tail` and `tail+1` atomically, and both contexts are released in
that same response cycle.  A zero-source K2 launch obeys the same atomic
two-entry rule.  This implements the atomic behavior assumed—but not proved by
RTL—in the M45-r2 transaction scheduler.

## Fail-closed contract

The engine makes protocol error sticky and disables all ready/valid endpoints
after an overlapping add/sub mask, illegal or duplicate launch, unexpected or
mismatched response, or positive/negative signed-19 overflow.  Reset clears a
saturated metadata FIFO, a stalled request, a stalled response, a full
completion FIFO, or a stalled output.  Context, response-metadata, and complete
FIFO occupancies are conserved by bound SVA.

## Verification boundary

The VCS test accepts K1 zero and long-burst traffic; K2 fully shared, partially
shared, and disjoint traffic; same-bank/different-row traffic; eight occupied
contexts; metadata and complete FIFO boundaries; random request, response, and
output backpressure; signed add/subtract including `-128`; and protocol and
overflow attacks.  A separate Python replay consumes only the accepted VCS
handshake ledger and reconstructs lowest-row union issue and signed-19x96
outputs.

M49 does **not** admit the M45 all-ten cycle projection, an external scheduler
or parent-DAG implementation, checkpoint/full-network equivalence, SRAM or
DRAM timing, synthesis/PPA/power/energy, system speedup, a 3x crossing, or a
Prosperity/Phi/DATE comparison.  The published M45-r2 95,047,672 integrated
cycle candidate remains transaction-level and its independent review still
requires the M45-r3 ready-scoreboard/metadata-capacity repair.
