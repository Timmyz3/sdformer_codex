# M691｜Table-A production Synopsys registry r9 author handoff

## Author verdict

`AUTHOR_GO_FOR_FRESH_INDEPENDENT_R9_HAMMER`，self-score `94/100`。

M691 保留 M671/M684/M687 全部只读，并以 additive r9 production proof 修复 M687
的四个 P1 和一个 P2。该 handoff 仅表明作者实现和 adversarial suite 已闭合；不等于
独立准入，更不等于 paper PPA ready。canonical 状态仍为
`production_runs=0 / authority=0 / bundle=0 / eligible=0 / headline=false /
analytical=false`。

## 修复内容

1. **真实执行形态门**：五个 tool snapshot 与 simv 必须是带 execute mode 的 ELF，
   六个 DB 必须是非 plaintext binary。每个 tool version 单独绑定 executable SHA、
   argv、log、exit 和 exact version。VCS 拆成 compile 与 simv run 两个 rooted step；
   compile argv 显式包含 design RTL、TB、SVA 三个 exact path和 simv output。
2. **十算子 scope 门**：冻结十个 operator→RTL module→instance anchor；top RTL 与
   mapped netlist 均必须非空，十个 anchor 必须同时出现在 RTL elaboration 形态和
   netlist hierarchy。builder 另将 manifest scope/design 绑定 M527 row mapping。
3. **SAIF/annotation 门**：直接解析 SAIF 的正 duration、timescale、top instance 与
   正 toggle；net/pin annotation coverage 均不得低于 95%；PTPX script 必须引用 exact
   rooted SAIF path。
4. **17 macro 与面积门**：netlist 必须包含 weight `8`、state `8`、parent `1` 的 exact
   macro reference/instance；同一 17 实例进入 PTPX census。只准
   `DC_TOTAL_INCLUDES_MACROS`，并要求
   `DC total = logic + memory-compiler macro area`，输出 logic+macro 不再双计。
   parent scratch 继续严格为 `1R1W`，macro-rounded total 为 `245760 B`。
5. **负值门**：所有导出的 power component 只要 `<0` 即拒绝，不再保留 r8 的
   `1e-12` 负 residual 窗口。

## 验证

- Python 3.6.8 compile：PASS。
- 作者 adversarial suite：`10/10 PASS`。
- 完整 synthetic grammar fixture：PASS；它明确是 non-authoritative，仅用于证明正路径
  可执行。
- plaintext tool/DB、缺 SVA argv、缺 vcs_run、wrong version、empty selected slice、
  zero/wrong-top SAIF、annotation<95%、缺 macro、PTPX 17-instance drift、area double
  count、`-5e-13 mW` power 均被拒绝。
- canonical CLI：
  `M691_REGISTRY_PASS production_runs=0 authority=0 bundles=0 eligible=0 headline=false analytical=false`。
- 未运行 EDA/GPU/remote/training/capture/performance。
- M686 与 docs/359 未修改；docs/359 SHA 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## Fresh-hammer boundary

独立评审仍必须攻击：ELF/DB media spoof、wrapper log consistency、compile→simv output
root、scope anchor替换/参数化实例、SAIF parser root/coverage、17 macro role swap、DC
area inclusion equation、严格负 residual，以及 canonical zero 不被 future proof 单独
升为 Table-A/headline。任何 P1 均应保持 NO-GO production admission。
