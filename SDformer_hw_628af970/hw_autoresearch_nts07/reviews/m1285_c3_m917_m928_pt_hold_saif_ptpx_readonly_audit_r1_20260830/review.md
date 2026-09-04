# M1285｜C3 M917/M928 Fixed-T10 PT hold→SAIF/PTPX 只读收口审计

日期：2026-08-30  
模式：独立、只读生产证据、fail-closed；未运行 EDA/VCS/GPU/远端，未修改 RTL  
裁决：**GO 准备唯一 additive PT/门级 SAIF/PTPX 链；STOP 把 M917 写成 hold-closed；STOP 用单个 Fixed-T10 功耗包装 C3 加速或节能。**

## 1. 结果身份与当前物理边界

M917 canonical result 与 M928 独立打铁的双层 seal 已重新执行 `sha256sum -c`，全部
通过。当前 C3 唯一合法的 mapped identity 是：

| 对象 | SHA256 / 身份 |
|---|---|
| mapped Verilog | `4618c2a8d90e952982f2bcdb1b24469c4b2fa08d42bc560d087d6d6e4a734750` |
| mapped SDC | `cdc5dcd0ac76b40428d2bbbcf7ea77d395a594ea378c5a44c37f0136197ca3a9` |
| SVF | `c1d5ed771e41d572322124775c25f2baf3aad09e853350e14c5eaebb4ea169cf` |
| DDC | `131c50b3c5fea9f9bfe26058fe66468070d446b33c32f5cbf79c37ba50a92559` |
| slow/max DB | `79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af` |
| fast/min DB | `a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a` |
| TT power DB | `d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070` |
| foundry gate Verilog | `3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a` |
| PrimeTime | W-2024.09-SP3, `afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef` |
| VCS | V-2023.12-SP1, `0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287` |

SVF 只服务 Formality 映射证明；DDC 只适合由 Design Compiler 重新打开。PrimeTime
应读 mapped Verilog、mapped SDC 与冻结 NLDM DB，不能把 SVF/DDC 冒充 STA 输入。

M928 准入的物理点仍为 28 nm、3.000 ns ideal clock、ZeroWireload、0 macro：
`62,433.503388 um^2`，71,898 cells，setup 最差已报告 slack `+0.0003 ns`。
但 DC 的 hold 诊断为约 `-0.02 ns` worst、`-58.19 ns` total、9,741 violations，且
M917 没有 hold optimization 或独立 PrimeTime。因此“setup/area admitted”与
“hold/full STA closed”必须继续分列。

## 2. PT 能否在不改功能 RTL 的情况下完成 hold

答案分两层：

1. **PT setup/hold 报告可以**直接在 M917 精确 mapped netlist 上完成，无需改 RTL
   或网表。合法 recipe 已在 M441 使用：`set_min_library slow -min_version fast`，
   读冻结 mapped SDC 后，用 OCV 明确绑定 slow/max=`ssg0p9v125c`、
   fast/min=`ffg1p05vm40c`，分别报告 max/min。该动作只增加独立诊断权限。
2. **M917 原网表的 hold closure 不能靠重新报告获得。**若 PT 的 min slack 为负，
   必须产生一个新的 hold-fixed mapped identity（DC `set_fix_hold` + incremental，
   或合法 PrimeTime ECO 插入 delay/buffer），随后做 RTL↔新网表 Formality，并重新跑
   setup/hold。功能 RTL可以保持不变，但网表、面积、功耗与 seal 必须全部换身份。

因此本审计对“PT 报告”给 GO；对“M917 已 hold closed”给 STOP；对“零 RTL 改动、
但新网表 hold ECO”给 CONDITIONAL GO。即使新网表在 pre-layout 下闭合，没有 SPEF、
CTS 与 routed interconnect 时仍只能称 pre-layout STA，不能称 signoff。

## 3. 现有活动入口能复用什么

M518 r11 VCS 证据 seal 已重验通过：功能 campaign 为 0 assertion failure、0 numeric
mismatch，N1/N4 accepted-start-to-retire 为 29/80 cycles，核心 issue 为 17 cycles/tile。
但它**不是可直接使用的门级活动入口**：

- TB 没有 `$dump*`、VCD、FSDB 或 SAIF 生成；
- TB/SVA 大量层次引用 `u_dut` 的 RTL 内部信号；综合后这些名字被重写或优化；
- r11 的异常攻击、内部 tuple ledger 与 SVA 适合功能证明，不适合作为门级功耗
  测量窗口；直接删除失败检查后继续跑会造成 vacuous activity。

可以复用的是事务生成/oracle、公共 ready-valid/result 接口、3.0 ns clock，以及
N1/N4=29/80 的服务边界。门级活动必须新增一个 public-port-only TB/adapter，实例化
精确 mapped module，只通过公共端口完成至少 clean N1、clean N4、rail/random
context 和输出数值检查。SAIF 方法可复用 M438 已验证的 UCLI gate-only recipe：在
reset/config warm-up 后仅对 gate DUT scope `power -enable`，结束后 `power -report`。

## 4. 唯一 additive publication DAG

后续不要再分叉出第二套 C3 power flow。唯一合法 DAG 是：

1. **Identity preflight**：校验 M917、M928、r11 VCS seals，以及 mapped.v/SDC、
   slow/fast/TT DB、foundry Verilog、PT/VCS、docs/359 精确 SHA；collision/resource
   gate 必须先过。
2. **PT-A read-only**：对 M917 mapped.v + mapped.sdc 做 slow/max 与 fast/min
   pre-layout STA，封 `check_timing`、coverage、setup、hold、constraints、libs。
3. **Hold decision**：若 setup/hold 都 MET，直接进入步骤 5；若 hold 负 slack，进入
   步骤 4，禁止把 PT-A 封成 closure。
4. **Netlist-only hold repair**：同一 RTL/SDC/库生成新 hold-fixed mapped netlist；
   Formality PASS 后重复 PT，要求 setup 与 hold 都非负。该步骤改变 mapped identity，
   所有后续 SAIF/PTPX 必须绑定新网表，不能继续引用 M917 网表哈希。
5. **Mapped-gate activity**：新增 public-port-only gate TB + UCLI；用 foundry cell
   Verilog编译当前终端 mapped netlist，公共输出 oracle 0 mismatch；SAIF scope 只含
   gate DUT，必须报告 annotation duration、measurement cycles、非零 toggle coverage。
6. **TT PTPX**：PrimeTime PX 读与 SAIF 完全相同的 mapped netlist/SDC，link TT DB
   并保留 slow DB 供 SDC 解析，随后覆盖 power corner 为 `tt0p9v25c`；`read_saif`
   后要求 100% net/leaf annotation、非零 toggle 门、`check_power` PASS，再报告
   total/internal/switching/leakage、hierarchy 与 pJ/cycle。
7. **Independent hammer**：重算身份、scope、时窗、annotation、PT/Power 数字与双
   seal；只将 Fixed-T10 component absolute power/energy 加入 component annex。

建议的唯一 additive source 名称为：

- `dc_handoff/scripts/run_ptsta_m1285_m917_fixed_t10_exact_sha.tcl`
- `tb_m518/m1285_m917_fixed_t10_mapped_gate_public_wrapper.sv`
- `tb_m518/tb_m1285_m917_fixed_t10_mapped_gate_activity.sv`
- `dc_handoff/scripts/m1285_m917_fixed_t10_gate_saif.ucli.tcl`
- `dc_handoff/scripts/run_m1285_m917_fixed_t10_pt_hold_saif_ptpx_exact_sha.sh`
- `dc_handoff/scripts/run_ptpx_m1285_m917_fixed_t10_tt0p9v25c.tcl`

这些都必须是新文件；不得覆盖 M917 netlist、r11 TB、M441/M438/M448 模板或现有
结果。若 hold repair 发生，里程碑必须升版，source contract 中绑定新的 mapped SHA。

## 5. 公平功耗分母与论文红线

当前不存在能支持 C3 “speedup”或“energy saving”的公平 power baseline。M917 只有
Fixed-T10 单点：它可报告绝对 mW、pJ/cycle，结合同一活动窗口可报告 component
energy；它不能单独产生比值。

M273 rank-3 不能直接作分母：它执行不同的 factorized arithmetic，ep35 checkpoint
未准入，且 matched PPA/activity 不存在。M265 的 `3.399935x` 也是 analytical、
非 area-matched 的 module-cycle 点，不能与 Fixed PTPX 相乘。

若未来要报 C3 相对功耗/能效，公平分母必须同时满足：同一 checkpoint 准入与质量
门、同一公共接口/clock/library/SDC/corner、同一 input transaction trace、同一
accepted/retired work、同一 SAIF 测量窗口、各自 mapped-gate 100% annotation，并
分别包含各自完整逻辑/存储。未满足前，论文只能写：

> Fixed-T10 exact component 在 28 nm pre-layout mapped-gate activity 下的绝对功耗/
> energy point；该点不代表 C3 加速、系统加速或相对节能。

禁止写成：Fixed power = C3 speedup；17 cycles/tile = 相对 rank-3 的物理 speedup；
`3.399935x × Fixed power`；或把 component pJ/cycle 升级成 mJ/frame/全系统能效。

## 6. 最终 GO/STOP

| 动作/主张 | 裁决 |
|---|---|
| M917 exact mapped netlist 的独立 PT setup/hold 报告 | **GO** |
| 在不改 RTL 的前提下生成新 hold-fixed 网表并做 Formality/PT | **CONDITIONAL GO** |
| 把 M917 当前网表写成 hold closed / STA complete | **STOP** |
| 新建 public-port-only mapped-gate SAIF source | **GO** |
| 在 exact source hammer、EDA 空闲后执行 gate SAIF/PTPX | **CONDITIONAL GO** |
| 单个 Fixed-T10 PTPX 作为绝对 component power | **GO，严格限定语** |
| 单个 Fixed-T10 PTPX 作为 C3 speedup/energy saving | **STOP** |
| M273 rank-3 直接作为当前公平功耗分母 | **STOP** |
| macro-inclusive、post-route、system/headline claim | **STOP** |

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
