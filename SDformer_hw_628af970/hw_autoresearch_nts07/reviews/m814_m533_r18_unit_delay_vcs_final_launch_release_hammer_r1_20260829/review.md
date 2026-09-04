# M821 — M814/M533 R18 final-launch release hammer

## Verdict

**PASS 100/100; P0/P1/P2 = 0/0/0.** The exact double-sealed M814/R18 release may be consumed once by the exact functional command below. This hammer did not invoke live VCS, simv, a license query, or any EDA tool, and it created no result or attempt.

## Evidence

- The request, release, runner, source contract, candidate, TB R8, M815 source hammer, M819 candidate hammer, R17 failure and M812 failure audit all pass strict identity and double-seal checks.
- Pinned Python 3.6 reran the TB R8 source-static check, closure positive and all three negative mutations. The runner has 32 custom functions, 244 conservatively enumerated call sites, zero undefined calls and zero duplicate definitions; all 20 external commands match the exact executable whitelist.
- The exact pre-mkdir stub returned rc86 with the five required events and zero VCS identity, license, compile, simv, result or attempt side effects.
- Wrong release/runner/candidate/M815/M819 identities, duplicate-key release/review payloads, and isolated existing-result/final-output collisions all fail closed.
- R17 remains permanently consumed `FAILED_DO_NOT_CITE`. R18 result and attempt identities were absent before this review.

## Authorized command

```bash
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1 VCS_ARCH_OVERRIDE=linux SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_m814_m533_m528_dead_write_only_1rw_unit_delay_r18_exact_sha.sh
```

Exactly one foundry-UNIT_DELAY functional VCS compile and one simv execution are authorized. Any raw result requires a fresh result hammer.

## Claim boundary

This is release-integrity evidence only. It does not verify the RTL, timing, cycles, speedup, PPA, energy, full network or a paper claim. The upstream 435,293,339-cycle / 1.746753x CPU-ledger point and 240 KiB capacity statement remain unpromoted.
