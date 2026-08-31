# M1305 — independent M1304 VCS result hammer

## Verdict

**GO as a directed K1 diagnostic RTL functional VCS receipt; 96/100.**

The unique M1304 run compiled the exact M1293 eleven-member absolute filelist with top `tb_m1293_c2_dual_dut_functional_reachability`.  Every compiled M1293 source matches its frozen source authority; the two inherited support modules are additionally pinned by this hammer.  The M1300 release seal validates, only one matching run namespace exists, and each log contains exactly one command.

Compile/link completed and produced executable `simv`.  Simulation reached normal `$finish` with exactly one token:

`PASS_M1293_DUAL_DUT_FUNCTIONAL_REACHABILITY classification=BOTH_CLEAN_FUNCTIONALLY_EQUAL req=48 result=6 done=1 first=4/58/64`

No fatal, assertion failure, X escape/coercion marker or failure token appears.  Input and output SHA receipts reproduce exactly.  Numeric shell exit-code files were not separately persisted; successful compile/link, normal VCS report/`$finish`, and the post-run output SHA receipt establish the completed artifact chain.  This is retained as a P2 receipt-quality caveat, not silently described as an explicit numeric exit receipt.

The result proves only directed K1 diagnostic functional reachability/equality.  It does **not** admit mapped functionality, K8/equal-bandwidth behavior, performance, power, energy, system speedup, PPA or a paper headline.  This hammer launched no second VCS run or EDA tool.

`docs/359_DATE终局冻结_20260813.md` remains SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
