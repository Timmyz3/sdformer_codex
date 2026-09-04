# M1267 true-author / M1266 runner-compatible release hammer

Decision: **PASS, 100/100, P0/P1/P2 = 0/0/0**.

This directory is the exact M1266 path/schema frozen into the M1265 runner. The
fresh different author is `/root/m1267_c1_r12_exact_byte_release_hammer`; M1266
was separately used by an unrelated read-only DATE evidence audit. The identical
`alias_binding.json` in both sealed directories makes that numbering alias
explicit and does not alter any source.

The independent source-only hammer passed 113 checks. It rejected corpus byte
drift, malformed/missing release pins, duplicate compile/simulation mutations,
timeout/quarantine/attempt-gate deletion, claim inflation, old TB/filelist
seepage, and non-fresh namespaces. It also binds the M1265a exact-TB
reachability PASS. No VCS, simv, EDA, GPU, or remote command was run.

Authorization is limited to one future M1265 compile and one simulation using
the four exact SHA pins of this sealed directory. No retry and no other EDA are
authorized. Success can establish boundary-only functional VCS evidence only.
