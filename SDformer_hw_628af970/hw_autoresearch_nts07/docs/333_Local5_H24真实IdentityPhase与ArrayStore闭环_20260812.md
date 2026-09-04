# Local5 H24 真实 Identity、Phase 与 Array Store 闭环

> 日期：2026-08-12  
> 范围：Local5 单个真实 H24 窗口的 identity-service、逐事件 RTL、Acc32 与验证归档  
> formal G0：**DENY**

## 1. 结论

H24 已从“资源模型允许尝试”升级为“单个真实窗口完成 RTL 功能闭环”。冻结身份为：

```text
sample2 / stage3 / block0 / window1 / H24
```

no-hold 基线和 hold2 候选均完成真实 Verilator 回放，345,600 个 Acc32 分别与软件整数
金参考逐元素一致。候选精确命中运行前冻结的 589,824 个 held-valid pair 和
1,179,648 个 `valid=1,ready=0` cycle。47,941,735 行候选 trace 已转为 source-only
Phase Array Store，并通过独立逐行展开和 10/10 篡改负例。

本轮关闭 H24 单窗规模、协议、数值、cycle-free state 全序和归档扩展性；不证明
formal G0、1,200-window 覆盖、架构性能或 ASIC PPA。

## 2. 运行前合同

运行前先用 H3/H6/H12 真实 RTL 计数校准结构公式，再冻结 H24。合同为
`results/local5_h24_phase_structure_contract_v1_20260812/contract.json`，SHA256：

```text
df47e8bf8db7ec6e526f63e0381722291f1b0646fb49ec426fddf512ff3b857c
```

| 项 | 解析公式或冻结值 | H24 |
|---|---:|---:|
| `tx_state` | `3H^2 + 43202H + 1` | 1,038,577 |
| `acc_state` | `28800H^2 - 28800H + 1` | 15,897,601 |
| `head_state` | `46157H^2 + 1` | 26,586,433 |
| baseline trace row | 运行前冻结 | 46,762,087 |
| candidate trace row | 运行前冻结 | 47,941,735 |
| candidate stall cycle | `H^2 x 32 x 32 x 2` | 1,179,648 |

结构公式是 `[rtl校准结构合同]`，不是形式化证明；其作用是阻止运行后修改期望。

## 3. 来源绑定

identity table 位于
`results/local5_identity_service_tables_sample2_h24b0_v4_20260812/`。

| 项 | SHA256 |
|---|---|
| task plan | `16a9ee0abdf95fd1517718cfa7df58fdc23e980a9884115b20ce025c1e2163af` |
| identity manifest | `9d12b507a2bfa32220c0becd9d80cc890dc956064d8b907c2a47d13d7ca6b589` |
| verification receipt | `05e598cf07af3ffa1a3f1b5836fa1864f3584a61c163fe54c5bffaeac96d23d2` |
| software expected | `d9e546df5287c377e1712630bd496732655982a57b4737b2f13200f2ce50aeae` |

runner 把 table、24 个 head vector、combined vector、权重和 expected 复制到只读
`source_snapshot` 后执行。运行前、运行后与 snapshot 的 41 项清单一致；最终主包绑定
14 项外部输入、156 项内部文件和 6 项源码。

## 4. 真实 RTL 和 Acc32

正式包：`results/local5_h24_identity_phase_array_canary_v3_20260812/`。

| 指标 | no-hold 基线 | hold2 候选 |
|---|---:|---:|
| trace row | 46,762,087 | 47,941,735 |
| relation pair | 259,200 | 259,200 |
| weight pair | 589,824 | 589,824 |
| final pair / Acc32 | 345,600 | 345,600 |
| held-valid pair | 0 | 589,824 |
| stall cycle | 0 | 1,179,648 |
| Acc32 mismatch | 0 | 0 |

基线、候选和软件 expected 的 Acc32 SHA256 相同：

```text
2c5836342f0b3abe0197122ac9ed003b58f086b2da84016f0d8f81bd8462a05a
```

候选 trace 与 Phase Store 展开后的 CSV SHA256 相同：

```text
096d4e0c6f6154cb80433d088a6355af941046749ed55f6c33da591e8ae56e9c
```

基线与候选验证周期为 30,538,802 和 31,718,450，但 v8/v10 release 不只存在 hold
单变量差异，禁止计算架构加速比。仿真 wall-time 和 RSS 也只用于验证资源管理。

## 5. 独立 state 结构 oracle

首轮独立复审发现 P1：原 trace verifier 只保存 RTL 派生的 state digest，没有独立冻结
state 全序。为避免把现有 digest 再抄成“参考”形成自证，本轮新增：

`scripts/verify_local5_identity_state_structure_v3.py`

oracle 不读取 RTL state 行来生成期望，只根据 `H`、T450、`HEAD_DIM=32`、
`OUT_DIM=32` 和控制合同流式产生：

1. 每 tile 的 `TX_WAIT_HEAD -> TX_RUN_HEAD -> TX_HEAD_DONE`；
2. 每个非首 head 的 14,400 次 `ACC_RMW_WAIT -> ACC_IDLE`；
3. 每 head 的 1,024 次权重、450 次 relation 和 14,400 次结果状态；
4. 三类 state 在每个 head/tile 内的全局交错顺序。

结果包：`results/local5_h24_state_structure_postreview_v1_20260812/`。

基线与候选各有 43,522,611 条 state 行逐行匹配，且 cycle-free 全序 SHA256 相同：

```text
20f134526627d821a18642c9a67b44e1654e2c9ecce583396ca24cc79376ca6f
```

四项单元测试通过，包括闭式计数、H1 完整全序正例和 state 值篡改拒绝。该证据关闭
“内部 state/phase 无独立结构 oracle”的 P1，但 cycle 仅检查单调性，仍不声称
cycle-sensitive 时序 oracle。

第二轮复审指出 identity 参数原先由 CLI 注入，且 H1 正例仍复用同一个生成器。为此又
补两道交叉约束：H24 报告必须由父 `complete.json` 同时绑定 identity 和 trace SHA；
H3 解析 oracle 必须与早先独立冻结的 state-reference 计数合同一致。两项均通过，结果
分别写入 H24 的 `parent_binding` 和 `h3_analytical_frozen_crosscheck.json`。这提高了
身份和 oracle 校准可信度，但 H24 全序的期望实现仍是当前解析模型，不能称形式化证明。

## 6. Phase Array Store

目录：`results/local5_h24_identity_phase_array_canary_v3_20260812/phase_array_store/`。

| 项 | H24 实测 |
|---|---:|
| typed array | 27 |
| expanded row | 47,941,735 |
| instance | 1,177 |
| unique payload | 1,194,625 |
| store bytes | 1,227,490,117 |
| legacy NPZ | 未生成，source-only |
| tamper | 10/10 拒绝 |

| 阶段 | wall-time | max RSS |
|---|---:|---:|
| generator | 1,625.84 s | 377,836 KB |
| verifier | 417.91 s | 407,636 KB |
| tamper regression | 118.18 s | 407,508 KB |

三者均低于 512 MiB，也低于运行前 20% 保护值。1.23 GB store 和 Python RSS 是验证
基础设施数据，不是片上存储、面积或功耗。

## 7. 事后归档修复

首轮复审另发现两个 P2：嵌套 `complete.json` 的 absolute path 带旧 staging 名，以及
verifier 边界文字硬编码“H24 尚未通过”。处理方式为：

- 不改写已密封原包；新增 `phase_store_relocation_receipt.json`，以当前位置相对路径
  重验三项 SHA，并绑定原 `complete.json`；
- live runner 后续统一写 `path_base=package_dir` 的相对 locator；
- live verifier 改为“其他窗口、formal G0 与 full encoder 尚未通过”。

外部不可篡改信任根仍未建立，继续保留为 P2。

## 8. 证据晋级

允许晋级：

- `[待验证] -> [rtl]`：一个真实 H24 窗口的 no-hold/hold2 协议和 Acc32；
- `[待验证] -> [独立解析结构oracle]+[rtl-trace-derived]`：43,522,611 条 cycle-free
  state 全序；
- `[待验证] -> [rtl-trace-derived]+[独立软件逐行验证]`：47,941,735 行 Phase Store；
- `[模型] -> [资源实测]`：该窗口 Phase Store 容量和脚本 RSS。

不能晋级：

- 100/100 sample numeric formal；后续 `docs/334`、`docs/336` 已把 numeric coverage
  推进到 15/100，
  仍未达到 formal G0；
- 462,600 条正式 phase ledger、1,200-window archive 和 `admission_receipt.json`；
- H24 的 Icarus 交叉复验与 cycle-sensitive 独立时序 oracle；
- full encoder、DC/STA/SAIF、ASIC PPA、系统 FPS 或架构性能；
- Local5 架构新颖性或 DATE 接收结论。

## 9. DATE 裁决

首轮独立 DATE 风格复审为 `4/5 Conditional Accept`、P0=0、P1=1、P2=3。它允许
H24 晋级为单窗口功能 `[rtl]`，但要求补独立 state oracle。解析 oracle 已关闭 P1，
并修复两个可本地关闭的归档 P2；外部信任根 P2 保留。

第二轮针对性复审仍为 `4/5`、P0=0，确认 P1 实质关闭、locator 和陈旧边界 P2 关闭；
同时要求明确 cycle-free 与 cycle-sensitive 边界，并指出测试独立性和 identity 绑定为
残余 P2。后两项已补父包绑定与 H3 冻结 reference 交叉检查；外部信任根仍未关闭。

最终针对性复核裁决为：`P0=0`、`P1=0`、`P2=1`、`4/5`。它确认 identity 绑定和
oracle 校准两个残余 P2 已在证据包层面关闭；唯一保留的 P2 是所有收据、源码和 SHA
仍在同一可修改存储域。准确表述应为“解析 oracle 经独立冻结 H3 RTL state reference
校准”，不能写成“存在第二套 H24 oracle”。

本轮显著增强 Local5 的 implementation credibility 和 scalability evidence，但不增加
architecture novelty。主包状态仍为：

```text
PASS_SEALED_H24_IDENTITY_PHASE_ARRAY_CANARY_NOT_G0
```

formal G0 保持 `DENY`。下一步应回到正式 1,200-window ledger/admission，而不是把单窗
闭环写成整网完成。
