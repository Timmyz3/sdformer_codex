# M799 / M533 R17 source-author handoff

M797 rejected R16 only because the mandatory dry-run harness used the Python 3.7+ text=True argument. R17 changes that argument to Python-3.6-compatible universal_newlines=True; after normalizing identities and the three test hashes, the R17 runner is byte-identical to R16 and retains all 76 literal SHA edges.

The source author executed the exact pinned Python 3.6.8 closure positive, all three negative mutations, and the runner-owned stub path. The stub returned rc86 with exactly five events and zero VCS identity, license, compile, simv, and result side effects. R16 remains unauthorized; R17 is source-only and launch remains false.

A fresh independent hammer is required. It must rerun every test and may not query licenses, run VCS/simv, author a release, or create a result.
