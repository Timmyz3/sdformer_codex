# M1029 M1022 license failure audit

M1022 is consumed and must not be retried.  Its Full64 frontend started
correctly, then stopped in `COMPILE_k1` with return code 255 because the caller
used `env -i` and removed both license-routing variables.  No simulator was
created, no gate case ran, and no SAIF file exists.

M1020's clean-environment `vcs -full64 -ID` smoke correctly closed the prior
`vcsMsgReport` path defect, but `-ID` does not compile a design and therefore
does not perform the required license checkout.  The additive repair must:

- preserve a caller-provided nonempty `LM_LICENSE_FILE` or
  `SNPSLMD_LICENSE_FILE` without printing or recording its value;
- compile a frozen tiny SystemVerilog source before consuming the production
  attempt;
- seal that preflight in an isolated namespace; and
- use a fresh M1033 attempt/result namespace behind an independent M1032
  hammer.

This audit authorizes source/release repair only.  It does not authorize M1033,
PT, PTPX, DC, GPU work, power, energy, or system claims.
