# M1557 ordinary regression of M1556

Verdict: **the declared input-fixedness preconditions are met for exactly one subsequent D0/call-0 diagnostic pilot covering the three non-product configurations**. Production, product configuration, automatic retry, and paper claims remain forbidden.

The regression pins commit `bf502765` and confirms under CPython 3.6.8 and Python 3.10.18:

- the executable entry accepts only `config`;
- module load captures `payloads/c000_s10_d0.positive.le.bitpack`, shape `(10,1,1536,15,20)`, and SHA `37208563...78a1c` as scalar closure values;
- the D0 compact plane is exactly 576000 bytes;
- construction closes the source descriptor and retains a `bytes` snapshot;
- modifying the temporary source file after construction changes neither `bit()` nor the snapshot SHA;
- author test, synthetic self-test, and preflight pass in both runtimes;
- product validation, pilot release, and production release remain closed.

This was an ordinary software/data-consistency regression under the assigned scope, not a security or bypass test. No actual pilot, request zero, production population, product configuration, GPU, SSH, RTL, or EDA was run.
