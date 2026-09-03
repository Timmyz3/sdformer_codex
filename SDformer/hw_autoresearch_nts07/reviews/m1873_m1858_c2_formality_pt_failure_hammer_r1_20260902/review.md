# M1873｜M1858 C2 Formality/PT 唯一失败独立审阅

结论：**审计 PASS（99/100），M1858 整体 production admission 仍为 FAIL_CLOSED；P0=0、P1=1、P2=0。K8 原始 Formality 报告确实成功，但只能保留为封存的诊断事实，不得作为 C2 论文/生产准入结果。M1858 已消费、`retry=false`，不得重跑。**

## 唯一运行与双封

- attempt latch 与 PID 2511659 failure quarantine 的 manifest/外层 seal 均独立校验通过；M1860 release、runner 与 `docs/359` SHA 均精确一致。
- namespace 中 canonical result=0、failure quarantine=1，无遗留 work 目录。
- 只有 K8 Formality 产物：`formality.rc=0`且 internal-complete marker 存在。K8 PT=0，K1X8 Formality/PT=0。
- 本 M1873 审阅没有启动 EDA、license query、retry、GPU 或远程任务，也没有修改前序证据、RTL、runner 或 `docs/359`。

## K8 原始 Formality 是真的成功

K8 建立了有效的 `ARCH_MODE=0` reference/implementation pair，SVF guidance **3798 accepted / 0 rejected**。`report_status` 明确为 `Verification SUCCEEDED`：

- 33,656 passing compare points；
- failing=0、aborted=0、unverified=0、unmatched compare points=0；
- passing/failing 表的 BBPin 均为 0；
- 已冻结的 8 条 `FMR_ELAB-147` warning 数量不变。

因此这不是设计不等价，也不是 Formality 工具返回失败。失败发生在工具证明结束后的 Python black-box gate。

## black-box parser 为什么误报

旧 regex 只看行首 `u|e|*` 和后续非零 `Instances`，不区分 `TECH LIBRARY` 与 `DESIGN LIBRARY`。它精确命中两处：

- implementation tech library：`e SNPS_BUSHOLD`, `Instances: 2 of 2`；
- reference tech library：同样的 `e SNPS_BUSHOLD`, `Instances: 2 of 2`。

两侧都只是 TSMC library 中 `BHDBWP35P140/C0` 与 `BHDBWP35P140#PWR/C2` 的对称 synthetic hold-cell internal。同一报告还显示：

- `m` 是 **Technology Macro cell (.db)**，不是 unresolved design module；每侧唯一非零 `m` 条目是 `ANTENNABWP35P140#PWR_FM_BBOX / pwrBB`；
- DESIGN LIBRARY 的 12 个 `e *` 条目全是 `Instances: 0`；
- 真正非零的 DESIGN LIBRARY `u/e/*` 实例数为 0，BBPin=0。

不允许由此泛化为“忽略 tech-library e 黑盒”。只能为这一个精确、双侧对称的 `SNPS_BUSHOLD` case 建立白名单，任何其他非零 `u/e/*` 仍必须 fail closed。

## 最小合法 successor

M1858 本身不能修复或重跑。仅修 parser 代码虽然足以修正这次 K8 的误报，但不足以补出从未运行的 K8 PT 和 K1X8 整轴，所以**必须新建 additive campaign 并重走完整两轴 Formality/PT**。新 gate 必须：

1. 按 TECH/DESIGN library section 解析，不再跨语义匹配；
2. 任何非零 DESIGN LIBRARY `u/e/*` 立即失败；
3. TECH LIBRARY 中仅允许精确的双侧 `e SNPS_BUSHOLD / 2 of 2` 路径集，不得泛化；
4. `m` 按 technology macro 处理；带 `*` 条目只有 `Instances=0` 才能忽略；
5. 继续强制 BBPin=0，且 failing/aborted/unverified/unmatched=0、passing>0；
6. 经 different-author source review 与 exact one-attempt release 后才能执行。

## 论文边界

K8 的 33,656-point raw equivalence 可作为“封存诊断事实”保留，但 quarantine 明确标记 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`，而且 K8 PT/K1X8 均缺失。因此不得在论文中声称 C2 Formality/PT 已闭合，也不得由 M1858 引出 setup/hold、PPA、功耗、能量或性能新数字。
