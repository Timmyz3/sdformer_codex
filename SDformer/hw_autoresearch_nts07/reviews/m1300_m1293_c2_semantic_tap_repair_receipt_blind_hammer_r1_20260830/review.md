# M1300 — receipt-blind hammer of M1293 C2 source repair

## Verdict

**GO one fresh RTL-only VCS release; 96/100; P0/P1/P2 = 0/0/1.**

M1293 closes all four M1287 blockers at source level.  This hammer independently checked the endpoint's 24-case four-state projection, the dual-DUT request/result/token reachability chain, class-aware functional comparison including all accumulator lanes, exact thirteen-tap non-coercing fanout, exact filelist, closed contract key sets and frozen upstream identities.

The exact M1287 attacks were replayed receipt-blind.  Removing request, result or token reachability, inverting the request compare, or moving a PASS token outside the guarded atomic block all fail.  Every existing or newly added performance/power/system claim promotion, open-world claim key and Boolean-as-integer attack fails.  Comment-preserved unconditional valid/payload gates fail.  X-to-zero and case-equality tap coercions fail.

Therefore the root author may run **one** fresh VCS compile/simulation using `date_m1293_c2_dual_dut_source_only_vcs.f` and top `tb_m1293_c2_dual_dut_functional_reachability`.  This does not authorize DC, PT, PTPX, SAIF, power, performance, mapped-functionality, K8/equal-bandwidth, system-speedup or headline promotion.  Even a VCS PASS proves only the directed K1 diagnostic axis.

## Receipt-blind scope

No M1293 author receipt was opened.  Checks consumed only source, test, contract, frozen upstream seals and synthetic in-memory mutations.  No VCS/EDA/GPU/remote/production work ran in this hammer.

`docs/359_DATE终局冻结_20260813.md` remains SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
