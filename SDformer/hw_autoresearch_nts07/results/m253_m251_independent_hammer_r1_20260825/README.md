# M253 independent hammer of M251 + M251r2

Score: **90/100**. Severity: **P0=0, P1=5, P2=4**.

Verdict: **GO for the corrected, scoped four-Conv simulator result.** M251r2
correctly revokes M251r1's `[-2032,2032]` range and replaces it with the exact
16-term signed-INT8 range `[-2048,2032]`. This still fits signed12, so all
cycle results remain unchanged. The correction rejects a wrong SHA before
output creation and a clean replay is byte-identical.

An independent implementation rehashed and unpacked all 40 packed support
payloads and 40 float-value payloads, reconstructed all Conv3x3 rows, and
replayed 51,840,000 16-bit partition vectors. It exactly reproduces the work
and cycle totals:

- Natural candidate versus bit-sparse: `1.5406415197256267x` vector work.
- WIDE144: `18.833088225777775x` versus dense and `1.5405574102292159x`
  versus bit-sparse.
- SHARED96: `15.072002904483751x` versus dense and `1.2328984754458552x`
  versus bit-sparse.

These are isolated four-Conv simulator values. The dense numbers mostly
include the trace's pre-existing `12.225647387x` natural sparsity, so every
use must also show the incremental bit-sparse comparison. PAFT-versus-control
hardware gain, integrated RTL cycles, matched throughput/area, energy, system
speedup, paper PPA and headline remain false.

The most important remaining evidence gaps are the absent raw M73 split inputs
needed to independently reconstruct zero leakage, the absent paired no-PAFT
running-BN trace, and the lack of an INT8/PWP numerical bridge plus executable
matcher/packer/service boundary.
