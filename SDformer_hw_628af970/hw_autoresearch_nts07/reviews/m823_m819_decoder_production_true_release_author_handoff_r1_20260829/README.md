# M823 M819 decoder true-release author handoff

This package contains only a one-way true release and its source-level validation record. It did not invoke the formal runner, production simulator, RTL, VCS, EDA, GPU, or any remote task, and it did not create the canonical M819 attempt, result, or failure directory.

The release is ineffective until a receipt-blind final release hammer returns exactly `PASS100_M819_TRUE_RELEASE__AUTHORIZE_EXACTLY_ONE_PRODUCTION_REPLAY` with P0/P1/P2 all zero. Even after that gate, only the root agent may execute the exact pinned command, once. Any raw result remains non-citable until a fresh result hammer passes.

The replay boundary is frozen at 40 M686 plus 120 M699 records, T10, A1/K1x8/K8, 96 lanes, 240 KiB, Acc24, 3 ns, and 192 B/cycle. D1 is charged but nonheadline; the only legal headline comparison is typed signed K8 versus equal-service K1x8.
