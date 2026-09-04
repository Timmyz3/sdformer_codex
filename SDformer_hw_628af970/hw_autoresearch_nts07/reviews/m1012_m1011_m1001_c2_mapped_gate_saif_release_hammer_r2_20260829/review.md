# M1012 independent release hammer

Verdict: `PASS_M1012_M1011_M1001_RELEASE_HAMMER_R2` — exactly one future M1013 VCS+SAIF attempt is authorized; no automatic retry.

The additive repair is internally consistent. M1011 pins the actual frozen M1001 source-contract SHA (`7afc4c093b...`), the frozen M1002 outer seal, and the exact new M1013 runner. Its JSON payload and two file sidecars verify. The runner preserves the frozen three-axis by five-case semantics, requires fresh compilation per axis, reuses no old `simv`, and writes only to a new M1013 result/attempt/work/failure namespace.

The old chain remains evidence, not authority. M1003 contains the wrong source-contract pin (`7afc4c095d...`), M1004 correctly records `STOP_M1004_M1003_SOURCE_CONTRACT_PIN_DRIFT`, and neither the M1005 attempt nor result exists. No old source, release, runner, or hammer was edited.

This hammer is static. It did not execute M1013, VCS, SAIF, PT, PTPX, DC, GPU or remote work. Authorization is limited to VCS mapped-gate replay plus SAIF creation; it does not authorize power, energy, performance or PPA claims.
