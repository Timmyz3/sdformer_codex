# M904｜C3 ATLIF/PSN 物理闭合第一性原理审计

日期：2026-08-29  
模式：独立、只读生产证据、fail-closed；未运行 EDA/VCS/GPU/远端  
结论：**保留 Fixed-T10 exact island；matched PPA 仍未闭合；TDA/MCM-96 在 DATE Accept 时间窗内 NO-GO RTL。**

## 1. 当前能安全写什么

M518 Fixed-T10 是一块真实、可综合的 exact RTL island。它计算 10 个 temporal
input 到 10 个 output、每 tile 16 lane 的全 10x10 变换，共 `1600` 个 signed-INT8
product。96 个 multiplier slot 的容量下界是 `ceil(1600/96)=17` cycle；r11
Synopsys VCS 不是只验证公式，而是实际闭合 `17 cycle/tile`、`17N+12` 服务式、
N1/N4=`29/80`，51 个 assertion label、25/25 required cover、0 failure、0 numeric
mismatch。

这个结论只能写成 directed component RTL behavior。M265 的 `3.399935x` 是相同
ingress/config bandwidth/result sink 下的 analytical Fixed/rank3 module-cycle 点；
rank3 多了一套 stage2 资源，面积不匹配，并且其训练 checkpoint/精度没有在 ep35
准入，不能升级成物理或系统 headline。

还有一个必须纠正的命名：Fixed-T10 是 dense/full 10x10 transform；M273 才是
rank-3 factorized candidate。现有 ep35 checkpoint 的合法硬件分母是 Fixed，不是
把 M273 换名成“rank-0 exact”。

## 2. 为什么 matched PPA 仍为 false

磁盘上没有“M518 DC r5”这一身份。M518 r5 是早期 VCS 修复链，最终因 SVA
语法失败隔离，与物理 PPA 无关。真正失败的 matched DC 是 r2：Fixed child
`rc=0` 且写了 Tcl terminal，但 170 个 runtime snapshot 中 16 个低于旧合同的
64 GiB CommitHeadroom，最低 54.186 GiB；monitor=`1`、runner=`42`，整个点被
quarantine，rank3 从未启动。因此 Fixed 中间面积和 timing 永久只能诊断，不能
引用。

旧 r2 Tcl 还执行 `set_fix_hold`、incremental compile 和 hold-only optimization，
所以即便没有资源失败，它也不是干净的 setup/area 分母。r4 已将两点拆成独立
one-shot，并冻结为一次 `compile_ultra`、0 incremental、0 hold-fix、0 hold report
的 setup/area-only flow。代价是边界必须说清：r4 DC 成功也只闭合 setup/area，
hold 仍须后续 PrimeTime，paired PPA 仍须两个 point result 各自打铁后另发 paired
admission。

截至本审计，r4 Fixed/rank3 的 canonical result、attempt 和 paired admission
全部缺席，所以 `matched_PPA=false` 是正确状态。

## 3. 现有 RTL 完整度

| 对象 | exact/synthesizable | VCS | 物理状态 | 当前身份 |
|---|---|---|---|---|
| M518 Fixed-T10 | 是 | r11 directed PASS | r2 只留下隔离诊断；r4 未跑 | 当前 ep35 合法 exact 分母 |
| M273 rank3 | 是 | component PASS | 有旧 logic-only DC；无 r4 matched point | 条件候选，不是 ep35 exact 替代 |
| TDA/PCTDA/MCM-96 | 否 | 无 | 无 | pre-RTL 想法 |

因此 C3 不是“没有可综合 island”，而是“已有 exact island，但缺合法 Fixed 物理
分母和 paired closure”。

## 4. TDA/MCM-96 门判定

本审计按预先给定四门逐项判定：

| 门 | 要求 | 当前证据 | 裁决 |
|---|---:|---|---|
| exact | 0 mismatch | 没有真实 45-context checkpoint output miter | FAIL |
| issue | <=10 cyc/tile | M714 pattern capture 未产生；无可执行 port schedule | FAIL |
| active state | <=24 KiB | 逻辑表+Acc=20,960 B 仅余 3,616 B，漏 transpose/control/output；resident-45 为 57,600 B unreplicated / 921,600 B 16-lane replicated | FAIL |
| throughput/mm2 | >=1.25x | 没有 admitted Fixed area，也没有 candidate DC | FAIL |

PCTDA 的面积下界已经足以快杀当前“16-lane replicated table”形态。按 M716 的
128x128 宏面积和 M711 的 20,480 B replicated table，光表就需要 10 个宏，约
87,583.61 um2。用 r2 的 66,778.24 um2 **隔离诊断值**只做乐观快杀，且完全不
计 detector/mux/broadcast/Acc/output/control，issue=10 的 full-service
throughput/mm2 上限也只有 N1 `1.005x`、N4 `1.173x`，低于 1.25x。这里的数字
不是可引用 PPA，只说明该形态不值得投 RTL。

MCM-96 也不能直接绕过这个问题。共享引擎要服务 45 张不同的 T10 matrix；“层
静态常数”不等于“一张全局静态常数图”。要么复制 45 张 adder graph，要么将其
做成 runtime-programmable shift/add/mux 网络，后者已经失去固定 MCM 的主要面积
优势。要在 10 cycle 内完成，还需每周期 16 lane 各自产生 10 个加权输出并更新
160 个 accumulator；当前没有面积、路由或 timing witness。

所以本轮结论不是“DA 永远不可能”，而是：**DATE Accept 时间窗内不再写
TDA/MCM RTL；只可留作 future work/附录的 pre-RTL 负结果。**

## 5. 一个且仅一个下一动作

等当前 C2 三轴 DC **完整终止**且共享机通过冻结的 collision/resource gate 后，
只执行已经释放的 **M518 r4 Fixed setup/area one-shot**（`M518_R4_POINT=fixed`）。
不要同时启动 rank3、TDA/MCM RTL、VCS、PT 或第二个 C3 point。

理由：Fixed r11 VCS 已经闭合，r4 source/release 静态链也已经闭合；当前真正缺的
是合法 Fixed area 分母。再修 source contract 或再跑 Fixed VCS 不会增加论文
权限。该单点即便 PASS，也必须先做独立 result hammer，并且仍不能宣称 hold、
paired PPA、power/energy、系统倍速或 headline。

`docs/359_DATE终局冻结_20260813.md` SHA256 保持：
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
