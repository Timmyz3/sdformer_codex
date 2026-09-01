# M1780 author attestation

M1780 is source-only. I did not launch VCS, `simv`, Design Compiler, PrimeTime PX, or a license query, and I did not create an attempt or canonical result namespace.

The source implements the B8 scheduling question at the C2 memory boundary. Both modes have the same eight-bank response interface, ordinary persistent LRU8 INT8 row cache, eight signed Acc24 contexts, signed-code storage, issue interface and commit work. The baseline traverses buffered work token-major; the candidate traverses the same work source-group-major. The candidate may reuse only a fetched weight row. Every token retains its own signed value and performs its own add/subtract into its own Acc24 context.

The default source explicitly contains 12,288 B of shared row data, 2,304 B of Acc24 context, 6,144 B of signed source FIFO, 24 B of context tags and 48 B of active bitmap before control. This is larger, not smaller, than the M1763 2,128 B incremental lower bound. The future physical run must price all of it; no screening or directed ratio is paper-admitted here.

The independent Python model proves the directed token-major and group-major orderings conserve 96 row accesses, 1,152 typed issue accepts, 18,432 signed products and 48 commits. It predicts ordinary-LRU8 misses of 96 versus 12 and 1,152 versus 144 eight-bank weight beats. These are test expectations only, not a hardware result.

A different author must perform M1781 source review before any launch release. Commercial VCS, same-coordinate DC area, mapped activity/energy and an independent result hammer remain mandatory. TSBG stays within the C2 contribution and is not a fourth novelty claim.
