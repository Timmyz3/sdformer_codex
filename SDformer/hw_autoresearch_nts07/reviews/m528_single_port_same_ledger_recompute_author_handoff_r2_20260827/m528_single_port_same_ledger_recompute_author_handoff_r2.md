# M528 single-port same-ledger recompute r2 author handoff

## Outcome

The source-only M528 r2 package is ready for a new independent static hammer. No production CPU analyzer/runner, EDA, GPU, or RTL action was taken. `docs/359` and the frozen analyzer are unchanged.

The sealed r1 admission is permanently **NO-GO** and must not be edited, reused, or treated as authority for r2. R2 uses a new runner, execution-contract identity, canonical directory, attempt sentinel, author seal, and admission schema/status.

## The only two behavioral repairs

1. **Successful terminal cleanup.** R1 copied `production_stdout.log`, `production_stderr.log`, and `resource_preflight.log` into `result`, leaving the originals in the work root. After `result` became canonical, `rmdir` therefore failed deterministically. R2 moves those three logs into `result`, rejects any unexpected work-root residue before canonical commit, records that canonical has committed immediately after the atomic move, removes the now-empty work directory, then reaches `m528_complete=1` and the success message. The failure trap may quarantine only pre-commit work; it never touches an established canonical directory.
2. **Commit headroom.** The independently reviewed floor changes from `67,108,864 KiB` (64 GiB) to exactly `50,331,648 KiB` (48 GiB). The static review bounded this frozen workload at 6 GiB committed memory, retaining an 8× guard. The `128 GiB` MemAvailable floor, `32 GiB` SwapFree floor, three snapshots, clean `failcnt/under_oom/oom_kill`, three workers, chunksize two, and UID-local Synopsys/VCS/simv collision checks are unchanged.

All other differences are revision identity needed to make r2 non-reusable with r1. The analyzer SHA, frozen inputs, governing contract, cycle/traffic/capacity computation, sample-major and operator-isolated distributions, output result schema and filenames, CPU decision gates, and claim boundary are unchanged.

## Independent reviewer request

Please perform a source-only review without invoking the production analyzer or runner. In particular:

- statically prove the normal post-analyzer path leaves only `result` in the work root before the atomic canonical move and reaches the final PASS path;
- check the failure trap cannot mutate a canonical directory after `m528_canonical_committed=1`;
- verify the only resource-floor change is `50331648 KiB` and all other resource/collision controls remain frozen;
- verify the old r1 admission cannot satisfy r2's schema, status, runner, execution, author-seal, or canonical identities;
- verify the analyzer and every result-semantic input remain exact-SHA unchanged.

If and only if that static review passes, a different step may create a new double-sealed r2 admission. This author handoff itself authorizes zero CPU production runs, zero EDA/GPU runs, and no RTL work.

## Claim boundary

There is still no M528 paper-admitted number. A future passing CPU run remains one H67 sequence and four bottleneck Conv3x3 operators only. It is not RTL, PPA, energy, a full-network/system speedup, or a DATE headline, and its raw output still requires an independent result hammer.
