# M1979 independent hammer of additive M1978 TSBG runner

## Verdict

**PASS — release authoring only.** P0/P1/P2 are all zero. This review authorizes creation of M1980; it does not authorize a license query, attempt, VCS, simv, DC, PT, or any result claim.

## M1975 repairs

| Gate | M1978 result |
|---|---|
| M1967 exact path | PASS: actual double-sealed basename includes `independent_load_handshake` |
| M1975 FAIL binding | PASS: exact review SHA `a0851bec...` is checked and sealed |
| Compile option diagnostics | PASS: rejects SVAA-RNF and ignored/unknown maxfail forms |
| Simv option diagnostics | PASS: same rejection set is applied after runtime |
| Compile/runtime placement | PASS: compile has only `-assert svaext`; simv has `global_finish_maxfail=1` |

The existing M1956 warning and ignoring lines are now both rejected. Eight order/case/unknown/ignored variants were attacked independently against each of the compile and simv regexes; all 16 were rejected. Nine native SVA/fatal/watchdog variants were also rejected.

## Future parser chain

The embedded parser consistently requires:

- M1979 schema/status and exact runner SHA;
- M1980 schema/status, one-license/one-compile/one-sim/no-retry budget, 180-second timeout gates, and exact identity containing M1972, M1975 FAIL, M1970 TB, and M1970 filelist;
- M1981 schema/status/severity and exact runner/review/release identity.

No future document can authorize this runner without matching the hashes supplied as runner arguments.

## One-shot and log acceptance

- Fresh M1978 attempt/result/failure/work/lock namespace is required.
- Same-UID EDA collision and 16-GiB memory/commit headroom gates run before launch.
- Attempt is consumed before the sole license query.
- There is one static VCS compile and one static simv execution; no retry path exists.
- GNU timeout directly wraps simv for 180 seconds, TERM then KILL after 10 seconds.
- Any nonzero or failed postcondition routes through `set -e` to double-sealed quarantine.
- Success requires one PASS, ten BEGIN/COMPLETE phase pairs, 52 load begins, 52 completions, zero timeouts, and no assertion/fatal/option diagnostic.
- Raw success remains `paper_admitted=false` and requires a different-author result hammer.

`docs/359` remains at SHA `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Required next gate

Create M1980 exactly as parsed, then obtain an independent M1981 release audit. Only that audit may authorize one M1978 attempt.
