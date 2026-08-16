# DP-TME端口感知架构修正与首版RTL验证

**日期**：2026-07-13  
**状态**：计算原语首版RTL通过本地验证；供数SRAM、packet adapter、真实定点和目标库DC未签核  
**软件语义**：`h = bias + weight[T,T] × x`，all-binary事件为`h >= threshold`

## 1. 本轮先纠正了什么

### 1.1 原34拍不是系统周期

旧整数模型证明了五路T2映射的代数等价，但没有表达物理输入和输出端口。若在两拍内并行处理5个空间位置，每拍需要5个不同的32-lane输入向量，而不是把一个`x_lane[32]`广播五次。

在8-bit输入代理下：

```text
T10：1 × 32 × 8 = 256 bit/拍输入
T2 G5：5 × 32 × 8 = 1280 bit/拍输入
T2 G5：每个2拍packet产生5 × 2 × 32 = 320 bit event
```

因此，34拍只在5个输入银行可并行读取、event packet能以至少约160 bit/拍持续排空时成立。若接单32-bit Router，81位置的5184个有效event bit至少排空162拍，计算打包收益全部消失。

### 1.2 减少pack组数不自动减少统一阵列面积

T10需要10个输出时间槽同时接收同一32-lane输入广播。为保持T10的810拍，物理阵列必须保留`10×32=320`个MAC。T2取G3或G4时，只能门控未用槽：

| T2模式 | 活跃MAC/物理MAC | T2计算下界 | 匹配出口 |
|---|---:|---:|---:|
| G3 | 192/320 | 54拍 | 128-bit |
| G4 | 256/320 | 42拍 | 128-bit |
| G5 | 320/320 | 34拍 | 256-bit |

若真正把阵列裁成`2G×32`个MAC，当前单输入向量广播映射下，G3/G4的T10都要跑两遍并增至1620拍，G1/G2更差。完整数据见`results/dptme_port_contract.md/json`。

这项修正使DP-TME从“固定五路最快”改为“完整320-MAC阵列 + 可配置T2活跃组数 + 端口匹配DSE”。G4是当前平衡接口候选，不是已签核答案。

## 2. 首版RTL结构

`rtl_hitflow/hitflow_dptme_array.sv`是一套时间复用计算阵列，不按81个ATLIF点复制：

```text
T10：group0的32-lane输入广播到10个output-time slot
T2：slot {2g,2g+1}选择空间group g的32-lane输入
每个slot/lane：signed multiply -> bias/accumulate -> threshold compare
最后一步：输出10×32 event packet、slot-valid、tag
```

默认参数为`LANES=32`、`SLOTS=10`、`PACK_GROUPS=5`、`X/W=8 bit代理`、`ACC=24 bit代理`。位宽都是RTL参数，不表示软件量化已冻结。

接口强制逐命令合同：

- T2必须恰好接受2个step；
- T10必须恰好接受10个step；
- 命令进行中`mode/group_valid/tag`必须不变；
- 提前`last`、超期、tag变化和`first&&last`均置`protocol_error`并拒绝握手；
- 输出反压期间event、hidden调试值、slot-valid和tag保持稳定。

RTL输出1-bit event。PyTorch二值ATLIF实际输出`event × threshold`，正式系统必须通过descriptor携带每site scale，并在下一Linear/Conv或gated-K中做逐位等价的scale folding；当前RTL没有证明该折叠的最终定点误差。

## 3. 验证结果

### 3.1 自检仿真

Icarus和Verilator断言仿真均覆盖并通过：

1. T10真实10步、group0广播、正负权重、bias和hidden累加；
2. T2两步、五组不同输入、三组有效尾packet和无效slot清零；
3. 命令中tag变化拒绝；
4. T10第二步提前`last`拒绝；
5. `first&&last`单步命令拒绝；
6. hidden和event阈值比较与testbench整数参考一致。

SVA验证输出反压稳定、`protocol_error -> !step_ready`、合法最后一步下一拍产生`out_valid`。

### 3.2 静态与通用综合

| 检查 | 结果 |
|---|---|
| Erie独立静态lint | 0 error、0 warning |
| Verilator RTL lint | 通过 |
| Yosys check | 0 problem、0 warning |
| 端口模型单测 | 3项通过 |

默认参数Yosys结构为3097个generic cells，其中320个`$mul`、321个`$add`、320个`$ge`、660个带使能寄存器及1425个mux。该统计证明结构展开符合320-MAC意图，但不能换算标准单元面积、频率或功耗。

## 4. 当前RTL还不能直接拿去做论文DC结论

当前模块是计算原语，不是最终DC顶层：

- `x_groups`是最宽1280-bit代理输入，尚未连接5-bank SRAM wrapper；
- weight、bias、threshold目前以展开端口输入，正式实现应由每site parameter SRAM/ROM广播；
- `out_hidden`是验证调试口，部署顶层应裁除或在验证wrapper中隔离，避免形成大路由负担；
- 320个乘法器尚未流水，500MHz能否达到完全未知；
- 没有饱和、舍入或显式量化，当前采用参数位宽内二补码运算；
- event packet还未连接128/256-bit serializer和Event Lifetime Router；
- 没有真实checkpoint逐site权重、bias、threshold和activation差分；
- 没有目标库DC/STA、SAIF功耗、门级仿真或Formality。

因此当前可声称“统一T10/T2映射已由整数模型和参数化RTL实现”，不能声称“34拍系统加速”“8-bit bit-exact”或“满足500MHz”。

## 5. 对架构创新的影响

DP-TME仍可作为候选，但论文贡献必须写窄：

> 面向同一网络内T10/T2 PSN时间矩阵的全尺寸slot阵列，通过编译期T2空间组映射、按组时钟门控和端口匹配packet排空，在不复制T10/T2物理核的前提下复用计算与控制。

不能把“时间并行”或“动态batch”本身写成原创。真正需要DC和trace证明的是：

1. 相对独立T10阵列+独立T2阵列，统一阵列总面积或EDP是否改善至少10%；
2. G4/128-bit是否比G5/256-bit在全encoder中有更低EDP；
3. 5-bank输入和宽event packet的布线/存储代价是否抵消34拍计算收益；
4. 编译期按site选择G3/G4/G5是否值得增加packet adapter控制；
5. 与LR-HTT直通结合后，宽packet是否能直接消费而不再次全物化。

## 6. 下一步

1. ordered profile完成后，用真实producer/consumer顺序重放G3/G4/G5与128/256-bit端口；
2. 写5-bank输入wrapper、双buffer event packet adapter和32-bit Router lane化接口；
3. 从H67/H68 checkpoint导出逐site定点向量，完成RTL逐命令差分；
4. 对乘法器增加可参数化1/2级流水，比较500MHz下面积和延迟；
5. 以G4/128和G5/256为首批同约束DC点，同时保留独立T10/T2阵列基线。

