# M2210 independent source review request

Review M2209 as a read-only, no-EDA source hammer. Do not run M2211 and do not modify M2197/M2199, the immutable parser, RTL, adapters, SVA, testbench, filelist, or `docs/359`.

Required checks:

1. Verify M2200 and this author receipt as exhaustive double-sealed inputs, including all source hashes and actual file modes.
2. Prove the only parser-launch change is the fixed regular nonsymlink executable `/opt/anaconda3/bin/python3.12 -B <immutable parser>`, with interpreter SHA/mode `873a.../0755` and parser SHA/mode `fde65.../0664`; reject direct execution, chmod, copy, or parser edits.
3. Prove the successful path removes `simv`, `vc_hdrs.h`, `csrc`, `simv.daidir`, and `simv.vdb`, rejects any remaining build item or symlink, retains logs/return code/receipt, then produces an exhaustive double seal.
4. Prove RTL, M803, M2018, SVA, TB, filelist, parser, VCS compile command, and simv command are byte/semantic identical to the frozen M2197/M2199 surface.
5. Independently reproduce rejection of the ten required runner mutations: direct parser, wrong Python path/SHA/mode, parser SHA/mode drift, missing `simv.vdb` cleanup, old result identity, retry, and old-artifact reuse.
6. Verify M2199 remains consumed, failed, non-citable, non-retryable, and unmodified. Verify the fresh M2211 result/attempt/lock are absent.
7. Verify `docs/359` remains unchanged and that no VCS, license, simv, EDA, GPU, or Git action occurred during M2209/M2210.

Only an exhaustive double-sealed M2210 result scoring at least 95 with P0/P1/P2 = 0/0/0 may authorize exactly one M2211 execution. Expected pass status:

`PASS_M2210_M2209_SOURCE_HAMMER__M2211_ONE_SHOT_VCS_AUTHORIZED`

The authorization dictionary must exactly be `{license_queries:1, vcs_compiles:1, simv_runs:1, parser_runs:1, all_other_eda_runs:0, automatic_retry:false, reuse_old_artifacts:false}`. M2199 retry and artifact reuse remain forbidden.

