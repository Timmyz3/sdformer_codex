# M1036 — cross-author hammer of the M1026/M1027/M1028 C1 full-replay chain

**Verdict: STOP and author an additive collision-gate repair.** Score 88/100; P0/P1/P2 = 1/0/0.

The M1025 authority, M1026 release and double sidecars, M1027 hammer, and execution-source receipt verify against their exact identities and seals. The runner is `f557c0e6…dc47`, release `96e9685b…1afd`, M1025 outer `7004ab97…43ff`, and M1027 outer `fc788926…057a`. Authority paths are hardcoded and the production M1028 result/attempt namespace is fresh.

The execution contract remains CPU-only and one-shot. Its frozen geometry is 51,840,000 rows = 10 samples × 4 operators × 432 partitions × 3,000 rows, with eight output blocks and three designs. Cleanup recursively seals failure quarantine. Capacity 214,912 B, matched cycles, and speedup all remain false pending a complete result and independent result hammer.

Six sandboxed faults—wrong runner pin, wrong M1025 outer, wrong M1027 outer, wrong release status, wrong release engine SHA, and occupied namespace—return before attempt creation. The sandbox replaces the production engine with `/bin/false`; neither the real M1028 runner nor the 51.84M replay is invoked.

One P0 remains: M1028 has no pre-attempt collision gate. A sandbox `pgrep` oracle reporting an active conflicting process was never queried. The runner created its attempt and reached the harmless replacement engine; cleanup then correctly sealed the failure quarantine. Thus collision does not fail before attempt as required.

Because M1028, M1026, and M1027 are already identity-sealed, the repair must be additive: a new runner/result/attempt namespace with a pre-attempt process/resource collision gate, a new release, and an independent hammer. Production M1028 itself remains unconsumed and must not run.
