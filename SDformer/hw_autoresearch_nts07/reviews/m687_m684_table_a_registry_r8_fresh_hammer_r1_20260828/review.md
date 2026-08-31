# M687｜M684 / M671 Table-A registry r8 fresh hammer

## 裁决

`PASS_CANONICAL_ZERO__NO_GO_PRODUCTION_ADMISSION__R9_REPAIR_REQUIRED`，评分
**58/100**，严重度 **P0=0 / P1=4 / P2=1**。

这不是对当前 canonical 状态的否定。当前仍严格保持
`authority=0 / bundle=0 / eligible=0 / headline=false / analytical=false`，不会把
任何 synthetic fixture、selected slice、0-macro 或 paper headline 误升格。NO-GO
针对的是 r8 声称已经闭合的“未来 production native Synopsys admission gate”：独立
fixture 证明它仍能把完全手写的证据链识别为 VCS PASS、Formality SUCCEEDED 和
rooted native extraction。

## 已核实且保留的修复

- M684 内外 seal、作者声明的 extractor/builder/tests/config/contract SHA 均复核一致。
- Python 3.6.8 对 extractor、builder、author tests 和本 hammer 编译通过；作者测试
  `8/8 PASS`。
- absolute、`.`、`..`、重复 `/`、反斜杠和中间 symlink 攻击均 fail closed；逐路径
  component-before-resolve 检查有效。
- VCS/DC/Formality/PT setup/PT hold/PTPX/memory-compiler 的 typed step 集、报告集、
  argv、script、log、exit、input/output hash map 和 component root 在“字节一致性”
  层面闭合。
- 三类 macro 的 organization、port、bank、instance 和 datasheet PVT 被严格比较；
  parent scratch 为 `1R1W`，总容量严格为 `131072+98304+16384=245760 B`。
- 0 logic area、明显负 PTPX component、负 setup/hold WNS 和 0 SRAM integrated power
  的拒绝路径存在。
- docs/359 保持
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## P1 findings

### M687-P1-01｜所谓 native tool run 仍是可手写的自洽语法环

`_expected_generation_argv()` 冻结的 VCS 命令只有
`vcs -full64 -sverilog -f <script>`，没有 `-R` 或独立 `simv` 运行，也没有 RTL/TB/
assertion 路径。七个 command script 又只要求未注释的
`M671_COMMAND_ROOT_BEGIN ... END` metadata block；它们不要求任何 `analyze`、
`elaborate`、`compile_ultra`、`read_verilog`、`read_netlist`、`read_sdc`、
`read_saif`、`report_power` 或仿真命令。这样的文件本身也不是可直接交给 VCS/DC/
PT/Formality 的有效 filelist/Tcl。

作者 fixture 使用纯文本 `M671_BINARY_TOOL_FIXTURE_*` 作为五个 executable、纯文本
`M671_BINARY_DB_FIXTURE_*` 作为六个 DB，再手写 scripts/logs/reports，完整
`extract_from_manifest()` 仍返回 VCS `PASS` 和 Formality `SUCCEEDED`。本 hammer
另行创建只含版本字符串和成功 marker 的两个独立日志，也分别通过两个所谓 native
proof parser。因此 SHA、argv 和 component-root 只证明这些字节彼此自洽，不证明
任一工具实际执行过。

修复门：production receipt 必须由只写 wrapper 产生，记录 cwd/env、真实
`execve` argv、PID、开始/结束时间、`/proc/<pid>/exe` identity 和 exit；VCS 必须拆为
compile 与 `simv` run 两阶段并分别封 executable/log/exit。工具可执行脚本与 wrapper
metadata 必须分文件；脚本应是有效 Tcl/filelist，并逐条引用已经 SHA-rooted 的输入。
工具版本必须来自同一 executable 的实际 `-version/usage` 子进程，而不是 receipt
字段。

### M687-P1-02｜full ten-operator scope 与 RTL/netlist 没有语义绑定

builder 只比较 `operator_scope_sha256` 是否等于冻结十字符串的 map SHA，并比较
`design_name` 字符串；extractor 只要求报告 header 复述同一个 design name。没有从
RTL elaboration、mapped netlist hierarchy 或 Formality compare-point census 派生
operator scope。

作者的 `design_rtl` 与 mapped netlist 都只有
`module h67_table_a_dense96_fixed_t10; endmodule`，但同一完整 extraction 被接受。
这直接证明只需把 manifest 中 scope digest 和 design string 改成期望值，selected
slice 或空 top 就能冒充 full M527；当前 canonical 的零 authority 会阻止它今天
升格，但未来加入 authority 后此语义缺口仍在。

修复门：冻结 exact source manifest/top；DC/PT/Formality 日志必须给出 top、link、
unresolved count、cell/reference/hierarchy census；将十算子分别映射到可审计的实例/
module pattern及非零实例计数；VCS 给出测试/assertion/coverage census；Formality 给出
reference/implementation top 与 compare-point count。scope root 必须由这些报告派生，
不能由 manifest 自报。

### M687-P1-03｜SAIF 被 hash 但没有 activity 语义或 annotation closure

`activity` 目前只经过 regular-file/media-label/SHA 检查。作者 fixture 的完整 SAIF
只有 `(SAIFILE (SAIFVERSION "2.0"))`，仍可驱动整条 extraction 返回正的 SRAM/总
功耗；PTPX parser 只解析最终表格，不要求 `read_saif`、duration、instance root、
strip path、annotated nets/pins 百分比或 zero/unannotated activity 报告。因而当前
dynamic power 证据可以完全脱离 VCS workload。

修复门：解析并冻结 SAIF duration/time unit/top instance、activity population 和
非零 toggle census；PTPX command/log 必须证明消费 exact SAIF SHA，并封
`report_activity_file_check`/annotation coverage。coverage 不足、duration=0、top/root
不匹配或 memory/clock/reset 未注释都必须拒绝。

### M687-P1-04｜macro inventory 与 netlist/功耗/面积尚未物理闭合

245760 B 与 `1R1W` 的 manifest/datasheet 算术是正确的，但三份 macro report、六份
SRAM DB、mapped netlist 和 PTPX `memory` group 之间没有 cell-name/instance-count
交叉证明。空 netlist 配合手写非零 `memory` power row 已经通过作者完整 fixture，
所以 17 个目标 macro 是否真的 link、实例化并被 PT/PTPX 计入仍未知。

面积还有独立口径风险：DC `Total cell area` 被直接命名为 `logic_area_mm2`，随后
builder 强制 `total_area = logic_area + datasheet macro area`。若 DC Total cell area
已经包含 linked macro area，就会双计；若 macro 在 DC 中为零面积 blackbox，则相加
才正确。r8 没有 hierarchical/reference area report 来裁定是哪一种。独立 fixture
展示该公式把 DC 0.60 mm² 与 datasheet macro 0.34 mm² 固定相加成 0.94 mm²，但没有
任何 macro-exclusion 证明。

修复门：封 `report_reference/report_cell -hierarchical`，要求 exact macro reference
和 `8+8+1=17` 实例；DC/PT/PTPX 都必须使用与 datasheet macro identity 对应的 DB；
netlist 中必须存在对应 cell/ports。面积明确选择并证明一种口径：要么 DC total 已含
macro、不得再加；要么从 DC total 精确扣除报告中的 macro area，再加独立 datasheet
area。PTPX memory group也必须以这 17 个实例 census 交叉闭合。

## P2 finding

### M687-P2-01｜direct extractor 的负残差有 1e-12 容忍窗

SRAM component 允许比 chip component 大至 `1e-12`，之后只在 residual
`< -1e-12` 时拒绝。本 hammer 构造出 `logic_internal=-5.000166947155549e-13 mW`
并通过 direct parser/guard。后级 registry `_number(..., zero_ok=True)` 会拒绝此值，
因此当前不会 admission，但与合同“negative component rejects”不完全一致。r9 应在
导出前将任何负 residual 严格拒绝；若需处理打印舍入，应先按报告分辨率做显式、可审计
的 reconciliation，而不是输出负数。

## 准入结论与下一门

M684 可以作为“canonical zero 仍安全”的方法学中间件保留，但不能成为 production
authority，也不能把 r8 的 future bundle 路径称为 native Synopsys closed。r9 至少
修完上述四个 P1，新增以下 adversarial tests 后才能请求 fresh hammer：

1. plaintext executable/DB、手写 marker log、metadata-only script 必须拒绝；
2. 无 `simv/-R`、无实际 source path、非零 exit 或 tool-child 缺失必须拒绝；
3. empty/selected-slice top、十算子任一实例缺失、macro count 非 17 必须拒绝；
4. header-only/zero-duration/wrong-root/low-annotation SAIF 必须拒绝；
5. macro area inclusion/exclusion 两个 fixture 只允许各自正确公式，禁止双计；
6. 任意负 logic/SRAM component，包括 `-5e-13`，必须拒绝。

本评审未运行 EDA、GPU、remote 或性能任务，未修改任何目标文件。
