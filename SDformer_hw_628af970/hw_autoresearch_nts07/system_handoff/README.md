# H67 ep35 system-trace handoff

This package is the data companion to the Git repository. It is deliberately
kept out of normal Git history because it contains a 591 MB checkpoint, a
roughly 958 MB Q/K trace population, and a large profiler JSON.

## What the package contains

- the frozen H67 ep35 checkpoint and deployment config;
- 1,200 Q/K trace NPZ files: 100 samples x 12 attention blocks, one window;
- the aggregate 100-sample hardware profile and CSV summaries;
- float and hardware-order valid825 receipts for the same checkpoint;
- the locked trace/profile entrypoints and current operator source needed to
  audit identity;
- a relative-path manifest with SHA256 and size for every packaged file.

The existing trace is attention-complete for the stated population. It is not
a full-network transaction trace: projection outside the attention slice,
ATLIF temporal-accumulator traffic, residual/FFN, decoder, DMA, and off-chip residency still
need to be exported by the full-network trace instrumentation.

The profiler now emits `execution_trace.csv` when run with `--ordered-trace`.
That optional file unifies operator, ATLIF, and attention call order per sample.
The sealed archive predates this field, so it must not be inferred from the
aggregate CSVs; a future GPU profile rerun is required before address-timed
memory generation.

The trace manifest locks the profiler and trace-writer SHAs. The included
`bsa_attention.py` is the current backward-compatible superset, not a claim
that its whole-file SHA equals the historical trace-time source. The H67 mode,
checkpoint, config, and emitted NPZ files remain independently locked.

## Build and verify

From the repository root:

```bash
python3 hw_autoresearch_nts07/system_handoff/scripts/build_h67_ep35_system_handoff.py --verify-only
python3 hw_autoresearch_nts07/system_handoff/scripts/build_h67_ep35_system_handoff.py --pack
```

On the destination server, verify the archive before extraction:

```bash
sha256sum -c h67_ep35_system_trace_handoff_20260821.tar.zst.sha256
tar --zstd -xf h67_ep35_system_trace_handoff_20260821.tar.zst
python3 h67_ep35_system_trace_handoff_20260821/verify_handoff.py
```

Use `scp` or `rsync` for the archive. Do not commit the archive or checkpoint
to ordinary Git. Git carries only the scripts, contracts, and small evidence.
