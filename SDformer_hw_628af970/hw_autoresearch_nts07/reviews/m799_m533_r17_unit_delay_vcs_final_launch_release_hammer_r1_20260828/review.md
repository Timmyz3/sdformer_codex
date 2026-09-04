# M810 — M799/M533 R17 final-launch release hammer

## Verdict

**PASS 100/100; P0/P1/P2 = 0/0/0.** The exact double-sealed release may be consumed once by the exact functional VCS command below. This hammer did not query VCS or a license server, did not compile or run simv, and did not create an attempt/result.

## Evidence

- Exact release `fe6814cd...a661`, runner `4d1b0a94...e0fe`, source, candidate, M801 and M805 identities are live and double sealed.
- Independent M805 and runner-canonical M805 packages are byte-identical, including manifest and outer seal.
- All 76 runner SHA edges are live; pinned Python 3.6 closure positive and all three mutations pass their expected gates.
- Pre-mkdir stub returns 86 with the exact five-event sequence and zero VCS identity, license, compile, simv, result or attempt side effects.
- Wrong release/runner/candidate/M801/M805 SHA, duplicate-key JSON, existing result, and final-output collision attacks fail closed.
- R15 remains permanently withdrawn with no consumed attempt/result; R16 remains `FAIL_SOURCE_GATE` with no release/result.

## Authorized command

```bash
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1 VCS_ARCH_OVERRIDE=linux SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_m799_m533_m528_dead_write_only_1rw_unit_delay_r17_exact_sha.sh
```

Exactly one functional VCS compile and one simv execution are authorized. Any raw result requires a fresh result hammer.

## Claim boundary

This is functional UNIT_DELAY evidence only. Acc24, 240 KiB, 435,293,339 cycles, 1.746753x, timing, PPA, energy, full-network and paper claims remain unpromoted.
