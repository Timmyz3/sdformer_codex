# M256 PAFT-ep4 running-BN INT8 and Acc19 bridge

M256 loads the exact PAFT-ep4 checkpoint through the frozen H9 overlay path
(`210/210` overlay keys, `missing=0`, `unexpected=0`) and exports all four
768x768x3x3 bottleneck kernels as output-contiguous signed INT8 payloads.
All `21,233,664` weights and all emitted payload bytes are audited.

The largest checkpoint-specific per-channel `sum(abs(qweight))` is `215,301`,
so every local or signed-motion accumulation fits signed 19 bits
(`[-262,144, 262,143]`).  Thus the PAFT checkpoint retains the M241 Acc19
datapath instead of falling back to the generic 21-bit dense envelope.  The
universal 16-source PWP range remains the corrected `[-2048,2032]` signed12
contract.

Across all 40 exact running-BN source records, `92,160,000` raw Conv outputs
were compared.  All four per-layer INT8-dequantized Conv audits pass: normalized
L2 error is `0.002130`--`0.004164` and cosine similarity is
`0.9999913`--`0.9999977`.  This does not admit quantized valid825 accuracy,
scale/BN hardware, M241r2 integration, energy, system speedup or paper PPA.
