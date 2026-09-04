# M2200 independent M2199 failure-result hammer

## Verdict

M2199 is permanently failed, non-citable, and non-retryable. Its unique execution root cause is exact: after VCS and simv completed successfully, the runner directly executed the parser at line 133. The parser is non-executable (`0664`; equivalent to the task's `0644` non-executable class), so the shell returned 126 before the parser started. `parser.log` is empty and no receipt or canonical result exists.

Using `/opt/anaconda3/bin/python3.12` to import the unchanged parser bytes in a read-only diagnostic reproduces a complete parser PASS. This confirms the logs themselves contain the expected ledger and nonzero coverage, but it does not rehabilitate M2199.

## Diagnostic facts—not claims

- VCS compiled seven modules; simv exited zero and emitted the unique RTL PASS token.
- Both modes commit 72 exact results and perform 72 context/slice/tag/terminal/Acc24 identity checks.
- Both modes exercise partial refill, eviction, reorder, request/bridge/commit stalls, signed sources, terminal behavior, and zero-descriptor skip; all required counters are nonzero.
- Ordinary performs 588 scalar/refill requests, while selective TSBG performs 156; products are equal at 3,264 per mode.

These are only diagnostic observations. They are not an admitted RTL verification or performance result because M2199 never published and sealed a canonical receipt.

## Additional evidence defect

The attempt marker is exhaustively double sealed, but the failure quarantine is not. It contains 92 regular files and two VCS-created symlinks, and has neither `SHA256SUMS` nor an outer seal. The failure trap calls `seal_dir ... || true`, so the symlink rejection was ignored before moving the directory. This review seals a complete read-only hash/target snapshot for diagnosis; it does not make M2199 citable.

## Minimal successor

The attached M2209 recommendation preserves the exact parser bytes, RTL, M803, TB, SVA, and filelist. The only execution-semantic repair is to pin the regular executable `/opt/anaconda3/bin/python3.12` by SHA and mode, pin the parser's SHA and non-executable mode, and invoke `python3.12 -B <parser> ...`. `chmod`, parser copying, parser editing, artifact reuse, and automatic retry are forbidden.

M2211 remains unauthorized until a fresh M2209 source and exhaustive M2210 independent source review pass. Any successful M2211 canonical result must be exhaustively double sealed before M2212 review.
