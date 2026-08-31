# M251r2 fixed12 signed-INT8 range correction

M251 r1 wrote the symmetric range `[-2032,2032]`.  That is only exact when
`-128` is reserved.  Because M251 had not yet bound the PAFT checkpoint to
such a quantization contract, the correct full signed-INT8 range for a
16-term PWP is `[-2048,2032]`.

The corrected range still fits signed12 exactly; the negative extreme lands
on the signed12 rail.  Vector bytes, service cycles, DMA traffic and all M251
cycle results are therefore unchanged.  This overlay revokes only the old
exact range statement and preserves all existing non-system claim limits.
