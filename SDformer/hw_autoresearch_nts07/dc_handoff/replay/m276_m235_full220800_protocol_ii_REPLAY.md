# M276 M235 full220800 protocol/II exact-VCS replay

From the `hw_autoresearch_nts07` root, run:

```bash
bash dc_handoff/scripts/run_vcs_m276_m235_full220800_protocol_ii_exact.sh
```

The launcher is fail-closed: it refuses an existing result directory, checks
the exact SHA-256 identity of RTL, SVA, testbench, full vector corpus, upstream
M245/M246 seals, contract and `docs/359` before compilation, verifies the
nested vector manifest, and uses only Synopsys VCS V-2023.12-SP1.

After a PASS, verify the citation bundle from the repository root:

```bash
sha256sum -c results/m276_m235_full220800_protocol_ii_vcs_r1_exact_20260825/RUN_MANIFEST.sha256
sha256sum -c results/m276_m235_full220800_protocol_ii_vcs_r1_exact_20260825/RUN_MANIFEST.seal.sha256
sha256sum -c results/m276_m235_full220800_protocol_ii_vcs_r1_exact_20260825/SHA256SUMS
```

The result is scoped to the unchanged M235 finalized-moment coefficient
engine on the frozen H67 ep35 s10 corpus.  It corrects the old test-driver
interval from 10 to intrinsic standalone latency/II 8/9 and closes legal
request backpressure plus the existing fail-closed illegal-zero attack.  It
does not establish a new speedup, full dynamic BN, event equivalence, PPA or
system performance.
