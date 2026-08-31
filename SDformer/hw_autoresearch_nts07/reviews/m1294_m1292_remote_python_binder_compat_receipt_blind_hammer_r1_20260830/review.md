# M1294 receipt-blind hammer: M1292 remote-Python compatibility successor

## Verdict

**78/100. Exact reviewed-byte transfer: GO. One-shot production execution: STOP.**

M1292 cleanly preserves M1257's authority/candidate/artifact topology, eleven
pre-attempt snapshots, three fully sealed execution memfds, F1--F4 receipt
closure, exact claim booleans, O_EXCL attempt and no-retry behavior. The only
production-policy fields changed are `interpreter` and `python_version`.

The release gate nevertheless remains closed. The source binds the string
`/usr/bin/python3`, not the interpreter entity. A symlink is accepted, no
lstat/realpath/dev+ino/mode/size/mtime/SHA enters policy or the attempt digest,
and the child reopens that path after the one-shot attempt is consumed. The
hammer reproduced this gap. An interpreter replacement in that window can
therefore burn the sole attempt under bytes different from those preflighted.

## Findings

- **P0:** interpreter entity / prepare-to-exec TOCTOU is unclosed. Use an
  additive successor that pins the supplied remote entity and preferably
  launches the child through an already opened executable fd; at minimum,
  snapshot before attempt and revalidate immediately before launch, with the
  identity included in the attempt digest.
- **P1:** the positive capability test passes the version constant into
  `probe_current_runtime`; locally it labels Python 3.12.13 as 3.12.3. It proves
  memfd/seals/stdlib/compile capability, not the exact target version. Probe
  `sys.executable` and `platform.python_version()` internally.
- **P2:** none.

## Evidence boundary

Local author regression is 14/14 and the independent fixture hammer is 11/11.
Local `/usr/bin/python3` is Python 3.6.8; local `/usr/bin/python3.12` is 3.12.13
and passed memfd, all four seals, stdlib, sealed compilation and write-rejection.
Those facts are not remote authority.

The main line supplied a read-only remote observation: `/usr/bin/python3`
currently resolves to `/usr/bin/python3.12`, Python 3.12.3, dev 1048625, inode
1347357695, mode 0x81ed, size 8020928, mtime 1774292672, SHA-256
`e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7`, with
memfd/all seals available. M1294 did not connect remotely and does not promote
that observation to execution authority; the source must bind it.

No production artifact was read or written, no checkpoint was selected, no
attempt was consumed, and no remote/GPU/EDA action occurred. `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
