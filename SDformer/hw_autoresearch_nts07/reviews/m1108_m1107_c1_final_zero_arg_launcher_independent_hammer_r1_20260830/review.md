# M1108 M1107 final zero-argument launcher independent hammer

## Verdict

`PASS_M1108_FINAL_LAUNCHER_HAMMER__ROOT_MAY_RUN_EXTERNAL_READONLY_GATE_THEN_UNIQUE_BARE_COMMAND`

Score: **98/100**. P0=0, P1=1, P2=0.

The exact M1107 launcher, launcher contract double seal, author receipt,
M1102 semantic/atomic chain, M1104 independent authority, pinned Python and
`docs/359` all match their frozen identities. The launcher `main()` was not
called. No production attempt, result, work, lock or quarantine was created;
neither exhaustive preflight nor the full replay iterator ran.

## Attacks

The hammer rejected extra argv, wrong Python identity, contract/source-receipt/
author-receipt byte changes, authority file and directory symlinks, and every
new M1102 stale namespace class (attempt, result, lock, work and quarantine).
An old M1095 marker remains outside the additive M1102 namespace and correctly
does not authorize or block M1102. Caller `PYTHONPATH`, authority-like variables
and `LD_PRELOAD` are erased; the surviving environment is the six hardcoded
locale/path/tmp/Python variables only.

An unsealed partial work directory cannot publish, an existing quarantine name
rejects without creating a stage, and the static production path contains one
call each to consume-attempt, execute-full and publish. The attempt is consumed
before preflight/full replay; caught post-attempt failure calls quarantine once;
there is no retry loop and the lock is released in `finally`.

Generic work accepts exactly zero or values at least eight. Canonical work also
requires divisibility by eight. The 1..7 interval, bools, negatives and canonical
9..14 values all fail closed.

## One explicit external-trust-root boundary

M1107 derives `launch_wrapper_sha256` from its own bytes. The M1102 atomic
library validates that field as a lowercase SHA but does not know the reviewed
M1107 digest. Consequently, a byte-mutated but internally self-consistent
launcher is not rejected by the internal authority function alone.

This is not a contract violation: the M1107 contract explicitly requires M1108
to pin the exact launcher, contract outer seal and author receipt before the
unique command, and does not require an internally hardcoded M1107 digest. The
P1 is therefore closed by the sealed external tuple. The naked command by
itself is **not** authorized. Root must first perform a separate read-only
regular-file/no-symlink/SHA/namespace/resource gate against
`external_launch_tuple.json`, then immediately execute the exact bare command
below without changing any file or namespace between the two steps.

## Unique command

```bash
/usr/bin/env -i LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/usr/bin:/bin TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/pytorch310/bin/python3.10 -I /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/system_simulator/scripts/run_m1107_m1102_c1_work8_full_replay_zero_arg.py
```

This command is authorized exactly once only after the external gate passes.
It accepts no launcher arguments. Maximum attempts is one, automatic retry is
false, and every produced raw result remains speedup-unadmitted and non-citable
until an independent result hammer.

## Claim boundary

M1108 is a source/launcher admission only. It admits no matched cycle, speedup,
PPA, energy, RTL or paper result. `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
