# M54 K4-C16 exact response-metadata experiment

M54 extends the independently admitted M49 K2-C8 standalone semantics to a
finite K1..K4, C16 engine. It does not inherit M52 transaction cycles as RTL
cycles. Its evidence boundary is accepted VCS handshakes and an independent
Python replay.

The engine owns sixteen explicit 4-bit contexts. Allocation selects the lowest
free ID; a context cannot be relaunched while owned, and all K contexts are
released in the same cycle as an atomic completion push. A launch contains an
explicit valid count from one through four and all participating IDs must be
distinct.

Each active cycle selects the lowest remaining row in the union of all K
source masks independently for each of eight banks. The physical row is read
once and each destination independently adds, subtracts, or bypasses it. Every
accepted request enqueues a real entry into a sixteen-entry response metadata
FIFO: a monotonic response tag, valid context count, all four context IDs,
bank-valid vector, per-destination valid/subtract vectors, and last marker.
Responses must match the FIFO head tag, count, contexts, and bank-valid bits;
unexpected, stale, or mismatched responses set a sticky fail-closed fault.

Final and zero-source groups reserve K completion credits. One simultaneous
legal output pop can contribute one credit. The engine atomically writes K
complete signed-19x96 vectors at consecutive circular tail positions, advances
the tail by K, and releases all K contexts in that cycle. The VCS test reaches
context occupancy 16, metadata occupancy 16, completion occupancy 16, FIFO
tail wrap, and occupancy 13 plus simultaneous pop/push4.

This milestone admits no M52 cycle projection, external scheduler, full-network
equivalence, SRAM timing, synthesis, area, frequency, power, energy, system
speedup, DATE headline, or best-paper claim. DC is intentionally not launched.
