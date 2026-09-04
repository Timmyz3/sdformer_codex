# M35-r4 canonical descriptor-ID ROM 规格

## 结论

M35-r4 将 M35-r3 的运行时 `delta/valid/sign/shift` 配置面收窄为 4-bit
`descriptor_id`。ID 0..9 一一映射到 H67 epoch-35 checkpoint 的十个冻结
canonical NAF descriptor，ID 10..15 进入 fail-closed `protocol_error`。RTL 端不再
存在可表达任意 signed-power tuple 的输入端口。

这一改动关闭的是独立评审的配置成员性问题，不新增算术加速比。当前里程碑只有
type-strict Python 模型和静态 RTL 候选审计；尚未运行 VCS、DC、STA 或 Formality。

## 身份绑定

- checkpoint SHA256：
  `4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158`
- canonical descriptor JSON SHA256：
  `209d34c4df8d3babf2ad701ee6c1305b2be17eea8ac7cf2bb62d703c5d9caff7`
- RTL 内嵌的只读短指纹：`209d34c4df8d3bab`
- 完整 descriptor 表和序列化规则由
  `contracts/m35_canonical_descriptor_contract_r4_20260822.json` 冻结。

短指纹不是密码学硬件校验器。完整 256-bit SHA 由构建/审计流程校验；RTL 源码本身
再由后续 VCS/DC/Formality 输入 ledger 绑定。这样避免在每个配置入口增加一个大而
无性能收益的通用 tuple checker。

## ROM 映射

| ID | delta | UQ0.24 raw | canonical signed powers |
|---:|---:|---:|:---|
| 0 | 2 | 16777214 | `+2^1` |
| 1 | 15 | 16777201 | `-2^0 +2^4` |
| 2 | 1 | 16777215 | `+2^0` |
| 3 | 21 | 16777195 | `+2^0 +2^2 +2^4` |
| 4 | 110 | 16777106 | `-2^1 -2^4 +2^7` |
| 5 | 18 | 16777198 | `+2^1 +2^4` |
| 6 | 121 | 16777095 | `+2^0 -2^3 +2^7` |
| 7 | 144 | 16777072 | `+2^4 +2^7` |
| 8 | 97 | 16777119 | `+2^0 -2^5 +2^7` |
| 9 | 588 | 16776628 | `-2^2 +2^4 +2^6 +2^9` |

四个 slot 固定为 shift 严格递增、valid slot 在前、invalid slot 的 sign/shift 全零。

## 负测闭环

模型复现 M35-r3 的四 slot 数值空间：每个 slot 为 invalid 或
`+/-2^shift, shift=0..9`。十个冻结 delta 共有 3620 个旧接口可接受 tuple：

- 独立 r3 评审定义的非 canonical tuple：3577 个；r4 adapter 全部拒绝。
- 评审按去零后的有序项判断为 canonical，但含 hole/slot 放置差异的 tuple：43 个；
  r4 只接受固定打包的 10 个 ROM row，额外拒绝其中 33 个。
- 因而 r4 compatibility adapter 在这 3620 个 tuple 中只接受 10 个，拒绝 3610 个。
- 首个复现 witness 为 delta=1 的 `[invalid, invalid, -1, +2]`，r4 拒绝。
- raw adapter 还类型严格地拒绝 invalid slot 非零 metadata、整数冒充 bool、float
  冒充 shift、重复 key 和非冻结 contract path。

## 算术与吞吐目标

每个输出仍计算

`Acc * (2^24-delta) = (Acc<<24) - sum(sign_k * (Acc<<shift_k))`。

模型对十个 descriptor、每个 7 个 signed32 边界值和 10000 个确定性随机值，共
100070 个 product 做了 exact identity 检查，0 mismatch，全部落在 signed56 范围。
静态 RTL 候选保留 8 outputs/packet、两级 elastic pipeline、无整数乘法运算符；
8 outputs/cycle 和 II=1 目前只是设计目标，必须由新 VCS trace 证实。

## 通用性取舍

这是 checkpoint-specific deployment ROM，不是通用 runtime CSD 引擎。优点是硬件配置
边界天然只能表达十个冻结 descriptor，避免 30/40-bit payload 比较器及其面积、时序
扇出。代价是 checkpoint 或 threshold 变化时，必须重新生成 contract/ROM、刷新完整
SHA，并重跑 VCS、DC、Formality 和系统 admission。论文中应将其表述为模型编译期
specialization，不能表述为可接受任意阈值的通用加速器。

## 后续新思门

1. 新建隔离的 M35-r4 VCS TB/SVA；覆盖十个合法 ID、六个非法 ID、全 mask、连续 burst、
   output backpressure、配置 release/reload、reset-under-stall 和 observed-output ledger。
2. 用独立 Python miter 重放 VCS 实际握手输出，证明 signed56 exact identity、事务守恒、
   顺序和无重复/丢失，并封存相对路径输入输出与 exit receipt。
3. 在相同 28nm/2ns 逻辑模型下做 r3/r4 DC A/B，只能在 Formality 通过后比较面积和
   setup margin；预期低 checker 成本不是已测面积结论。
4. 将独立 hammer review/admission 的哈希接入下一版 M39/M40，之后才能引用于
   Local/Motion 集成性能。

## Claim boundary

当前允许：十个冻结 ID 的一一成员性模型、3577 个已知非 canonical tuple 的穷举拒绝、
额外 33 个 hole/order variant 拒绝、100070 个 signed56 integer identity、RTL ROM 与
contract 的静态一致性。

当前禁止：VCS 正确性、RTL II=1、综合面积/时序、Formality、Local/Motion 全系统加速、
PPA、功耗能效、存储系统、精度、外部加速器对比、DATE headline 或 best-paper 声称。
