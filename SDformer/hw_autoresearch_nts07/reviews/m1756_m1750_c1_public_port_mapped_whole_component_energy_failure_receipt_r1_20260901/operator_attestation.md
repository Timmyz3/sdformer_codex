# M1750 post-failure unsealed-build disclosure

After the sealed M1750 one-shot failure, the root operator mistakenly invoked the private unsealed build's `simv -help` once through a pipe that terminated early.

- It did not include `-ucli`, the frozen UCLI script, `+M1739_UCLI_SAIF`, or a SAIF output environment variable.
- It did not produce a SAIF, PTPX report, canonical result, or replacement failure receipt.
- It was outside the one-shot runner and is not counted as a canonical M1750 simulation.
- Because the invocation was unsealed and did not preserve a complete provenance log, the entire old private build is permanently `UNSEALED_DO_NOT_CITE_DO_NOT_REUSE`.
- M1757 must use a fresh namespace and fresh compilation; no binary or `csrc` member from the M1750 private build may be copied, linked, executed, or used as a cache.

This disclosure is an operator attestation. The filesystem checks independently establish that no M1750 canonical result or SAIF exists at M1756 authoring time.
