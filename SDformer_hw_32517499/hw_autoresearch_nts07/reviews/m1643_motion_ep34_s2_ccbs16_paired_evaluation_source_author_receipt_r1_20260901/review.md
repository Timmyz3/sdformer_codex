# M1643 author receipt — S2/CCBS16 paired evaluator source

Status: **PASS source-only; different-author M1644 review required**.

The M1643 source is an inert in-memory accounting kernel. It opens no M1624
payload and exposes no actual evaluation CLI. Synthetic regression is 17/17 on
CPython 3.6 and 3.10, with byte compilation passing on both runtimes.

The fixed mechanism is one uint16 bound for each padded 16-source × 16-output
decision block at positive epsilon. Epsilon zero bypasses the directory and
must reproduce baseline AEE and cycles exactly. A drop receives weight-fetch,
compute or psum credit only when its decision precedes all three resource
starts; credited quantities are derived from the paired baseline block ledger.

The future gate is conjunctive: overall paired ΔAEE ≤ 0.02, every sequence
ΔAEE ≤ 0.03, metadata ≤ 2% of the same blocks' baseline weight bytes, and
ratio-of-sums same-resource local cycles ≥ 1.15×. If TSBG is admitted first,
the baseline must already include TSBG. Component speedups may never be
multiplied.

No payload, GPU, DSE, RTL, EDA, production execution, release, performance
number or paper claim is authorized by this receipt.
