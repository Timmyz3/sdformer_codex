# M1666 independent M1656 result hammer

Status: `PASS_M1666_M1656_DECODER_ACTUAL_PREFIX_RESULT__PREFIX_ONLY_DIAGNOSTIC__NO_L3_OR_PAPER_CLAIM`.

The exact sealed M1656 result passes a different-author, read-only hammer. It is exactly Motion ep34 `D0/call0/module0/timestep0`, destinations 0 through 41 and output blocks 0 through 3, under `DENSE_TYPED_K8`, `BIT_EQUAL_SERVICE_K1X8`, and `BIT_TYPED_K8`. The three sessions are distinct and configuration-bound. Their request counts agree with the metric ledgers; the exact M1645/M1638 lineage applies a miter to every request and every destination and validates the final three-session bundle. Both packed commit hashes are common across configurations: `b96c56...` for the packed metric sequence and `43638a...` for the L2 final commit digest.

Independent integer-ledger recomputation gives 1,034,451 / 519,007 / 481,123 cycles and 160,607,232 / 103,295,680 / 102,870,416 modeled transaction bytes. Thus dense versus equal-service bit is **1.9931x** with **49.83%** time and **35.68%** byte reduction; dense versus typed K8 is **2.1501x** with **53.49%** time and **35.95%** byte reduction; typed K8 versus equal-service K1x8 is **1.0787x** with **7.30%** time and **0.412%** byte reduction.

These are valid only for this fixed 42-destination prefix diagnostic. The hammer did not reopen the payload, regenerate the address stream, rerun the one-shot simulator, or authorize L3. It therefore does not admit a full-decoder, full-network, energy, RTL, EDA, Table-A, or paper headline claim. The exact address and commit digests are pinned and structurally checked, not independently replay-regenerated.

The same hammer passes CPython 3.6.8 and 3.10.16 and rejects 20 semantic, identity, RSS, ledger, and claim mutations. The authoritative M1656 result was not modified.
