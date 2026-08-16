# 统一Adaptive CSR运行时双格式架构与RTL验证

> 2026-07-18后续更新：本文记录的是首字`PEEK`版本，已经被`docs/110_TypedSlotMetadata与IPD选择性驻留架构闭环_20260718.md`取代。当前主线在payload commit时解析并保存格式，replay不再重复`PEEK`；Adaptive与IPD-only descriptor residency已经通过同顶层真实trace验证。本文旧周期只保留为迭代历史，不再作为最终结果。

> 后续状态：`docs/107_PhysicallyStripped_Direct_RAW41投影基线_20260718.md`已补齐projection-slice边界的物理裁剪Direct基线；它不等于完整single-context或目标库物理基线。独立第三轮审稿见`docs/106_DATE审稿人第三轮_AdaptiveCSR后评估_20260718.md`。

## 一、结论

本轮把上一版“逐stage分别编译IPD32W或FADC24，再离线取较优结果”的架构上界，落实为一套单一可综合配置：

```text
head slot 64-bit word stream
            |
            v
      word0 magic peek
        /           \
  IPD32W decoder   FADC24 streaming decoder
        \           /
         统一term/event接口
                |
                v
  GateStack product + bitmap multicast + AccTile
```

该统一前端不改变H67数值语义，也不在线删除任何term或token。CSR格式由payload header决定；既有RAW41精确回退仍由route控制面选择。一个物理核可以逐head执行IPD32W、FADC24和RAW41，不再要求按stage实例化不同投影核。

四stage真实trace回放采用同一Adaptive CSR RTL配置，Verilator周期和为`197857`，相对现有GateStack trace bundle的`278388`为`1.407x`，相对IPD32W无驻留的`285765`为`1.444x`。这是已实现硬件配置的局部trace结果，不是整encoder加速比。

## 二、架构合同

### 2.1 配置参数

为保持已有接口兼容，顶层继续使用`CSR_FORMAT_FADC24`参数，当前取值定义为：

| 值 | 物理结构 | 格式选择 |
|---:|---|---|
| `0` | 仅IPD32W decoder | 编译期固定 |
| `1` | 仅FADC24 decoder | 编译期固定 |
| `2` | IPD32W与FADC24双decoder加统一前端 | 每个head按word0 magic运行时选择 |

参数名是历史遗留，后续冻结接口时宜重命名为`CSR_DECODER_MODE`；本轮不为命名进行大范围接口改动。

### 2.2 统一前端状态机

`gatestack_adaptive_csr_replay_decoder`包含四个状态：

| 状态 | 行为 |
|---|---|
| `IDLE` | 接收一次head decoder start |
| `PEEK` | 接收并寄存word0，根据低16 bit magic选择子decoder |
| `START` | 向被选子decoder发start，未选decoder保持静默 |
| `RUN` | 先重放寄存的word0，再无损转发剩余word；输出只来自被选decoder |

首字窥探的真实成本没有隐藏：固定IPD路径在S0到S2分别出现约`0.7%`、`4.1%`和`1.3%`的周期损失。S1工作几乎为空，所以固定启动开销占比最高。

### 2.3 统一接口

两个子decoder共享以下输出合同：

- descriptor：`tag + term_count`；
- term：`gate_code + K lane + destination_count + term_index`；
- event：每拍最多4个token id、term/head边界标志；
- done：`tag + error`；
- 全部通道使用valid/ready回压。

后端只看到统一term/event语义，不需要知道源格式。FADC24的list/bitmap差异也在decoder内部消解，因此product engine、destination bitmap、multicast和accumulator均被复用，没有复制后端。

### 2.4 RAW41边界

Adaptive CSR只负责CSR内部的IPD32W/FADC24选择。RAW41仍是控制面根据`payload_mode_is_csr=0`选择的第三条精确路径。这种两级决策避免让magic检查承担RAW长度和异常恢复语义：

```text
payload mode
  |-- RAW  -> RAW41 decoder
  `-- CSR  -> Adaptive CSR -> IPD32W或FADC24
```

## 三、真实Trace结果

### 3.1 四stage同一硬件配置

证据：`results/gatestack_adaptive_csr_fulltop_20260718/report.{md,json}`。

| Stage | 实际header格式 | 周期 | 相对IPD无驻留 | 相对GateStack | terms | mismatch/protocol |
|---:|---|---:|---:|---:|---:|---|
| S0 | IPD32W | 2473 | 0.993x | 0.968x | 186 | 0/0 |
| S1 | IPD32W | 1802 | 0.959x | 0.931x | 0 | 0/0 |
| S2 | IPD32W | 22751 | 0.987x | 0.939x | 1956 | 0/0 |
| S3 | FADC24 | 170831 | 1.517x | 1.481x | 12888 | 0/0 |

四stage合计：

| 配置 | 周期和 | Adaptive相对值 |
|---|---:|---:|
| GateStack-IPD，含既有residency | 278388 | 1.407x |
| IPD32W，无residency | 285765 | 1.444x |
| 统一Adaptive CSR，无residency | 197857 | 1.000x |

该结果关闭的是“最强数字来自两个不同编译配置”的问题。它没有关闭单窗口代表性、descriptor residency、完整encoder和目标PPA问题。

### 3.2 同一context逐head交错

为证明选择器不是只在stage边界工作，新增两个S3、24-head同context用例：

| 用例 | 构成 | 周期 | terms | mismatch/done/protocol |
|---|---|---:|---:|---|
| 双CSR加RAW回退 | 11 IPD + 12 FADC + 1 RAW | 263303 | 30960 | 0/0/0 |
| 双CSR，无RAW展开 | 11 IPD + 13 FADC | 167665 | 12888 | 0/0/0 |
| 全FADC参考 | 24 FADC | 170831 | 12888 | 0/0/0 |

第一个用例验证三条路径在一个context内连续切换且数值精确。其高周期来自一个RAW head把61个共享term展开为814个event，而不是选择器错误。

第二个用例隔离RAW成本，证明IPD/FADC逐head交错可在相同term数下运行；其周期比全FADC低约`1.019x`。当前交错规则只用于覆盖，不是已冻结的最优格式策略，不能把该差值外推为profile100收益。

## 四、结构代价

同一Yosys `proc; opt; memory -nomap`结构代理：

| 结构 | generic cells |
|---|---:|
| IPD32W decoder | 448 |
| FADC24流式decoder | 954 |
| 两者简单相加 | 1402 |
| Adaptive CSR统一叶模块 | 1496 |

统一选择、首字缓存、mux和计数聚合增加94个generic cells，约为两个子decoder之和的`6.7%`。该数字说明选择控制不是主要结构项，但不能替代目标库面积、关键路径或功耗。双decoder是否优于按stage静态实例化，必须在相同工艺、SRAM宏和调度约束下比较。

## 五、验证闭环

### 5.1 已完成

- 四stage同一Adaptive配置：Icarus与Verilator/SVA全部通过；
- 两个同context交错用例：双工具功能计数一致；
- 所有被测用例：32-bit accumulator逐元素零mismatch，done error、protocol error、abort均为零；
- Adaptive外部断言：descriptor/term/event/done在stall下稳定，event count与mask一致，mask为前缀，error粘滞；
- IPD32W和FADC24子decoder继续绑定各自专属SVA；
- 原single-context与原multihead回归通过；
- Adaptive RTL和修改后的multihead顶层Erie lint均为0 error、0 warning；
- 六个Adaptive full-top用例（四stage加两个交错context）的Verilator构建均为0 warning、0 error。

### 5.2 验证中发现并修正的问题

首次S0 SVA运行失败，但不是数据通路错误：统一前端误复用了FADC24的“descriptor、term、event、done四相互斥”断言，而IPD32W合法地允许descriptor与term流水重叠。修正方式是：

1. 保留FADC24子模块的专属相位断言；
2. 保留IPD32W子模块原专属断言；
3. 新建格式无关的Adaptive外部断言，不强制两个格式不共有的微时序。

这是验证合同修复，未放宽valid/ready稳定性、event一致性或错误粘滞要求。

## 六、DATE Claim更新

### 6.1 本轮后成立

- 已实现一个可综合的、逐head运行时IPD32W/FADC24选择前端；
- 既有RAW41精确回退可与两种CSR格式在同一context内工作；
- `1.407x`是一个单一Adaptive硬件配置在四个真实trace窗口上的周期和结果；
- 两个CSR decoder共享同一term/event后端，未按105个PyTorch模块实例化硬件。

### 6.2 仍不成立

- `1.407x`是完整encoder、每帧或系统吞吐加速；
- Adaptive CSR降低了目标芯片面积、功耗、能量或EDP；
- IPD/FADC格式由硬件在线估计成本并自动生成编码；当前编码由上游descriptor compiler确定，硬件只按header执行；
- FADC24在所有stage、block、window和样本上普遍更优；
- descriptor residency已经支持FADC24；
- output-tile-stationary已经通过head-major spill物理基线证明；
- H67候选INT8部署在valid825上保持精度；
- 完整H67 encoder RTL或ASIC已经签核。

## 七、审稿缺口变化

| 第二轮缺口 | 本轮状态 | 说明 |
|---|---|---|
| 混合结果是编译期离线oracle | **关闭** | 单一Adaptive配置完成四stage与同context交错回放 |
| FADC与RAW同context未覆盖 | **关闭** | 11 IPD + 12 FADC + 1 RAW用例通过 |
| 运行时选择控制和面积代价未知 | **部分关闭** | RTL与generic cell已有；目标PPA仍缺 |
| physically-stripped Direct | 未关闭 | 下一优先级 |
| head-major partial-sum spill | 未关闭 | 下一优先级 |
| 目标库DC/STA/SAIF/LEC | 未关闭 | 缺工具、库、PVT和SRAM宏 |
| 多样本、多block、多window | 未关闭 | 当前仍是四个首窗口 |
| valid825部署量化 | 未关闭 | 候选INT8仅完成RTL回放 |
| full encoder Amdahl与存储分账 | 未关闭 | projection slice不能代表整网 |

## 八、下一步冻结

下一阶段不再增加第三种CSR编码。优先级为：

1. 将已完成的physically-stripped Direct projection baseline扩展到目标库PPA边界；
2. 实现head-major partial-sum spill基线，量化output-tile-stationary避免的SRAM读写；
3. 扩大真实trace，直接采集逐term fanout和多窗口周期分位数；
4. 评估FADC descriptor residency，若收益小则冻结S3 no-residency合同；
5. 获得目标库后执行DC/STA、mapped SAIF、SRAM宏和netlist LEC；
6. 完成valid825部署量化与full encoder分账。

## 九、入口

- RTL：`rtl_hitflow/gatestack_adaptive_csr_replay_decoder.sv`；
- 顶层接入：`rtl_hitflow/gatestack_multihead_decoder_projection_top.sv`；
- 统一断言：`verif_hitflow/gatestack_adaptive_csr_replay_decoder_assertions.sv`；
- 向量生成：`scripts/generate_gatestack_adaptive_mixed_vector.py`；
- 回归入口：`sim_hitflow/run_gatestack_adaptive_csr_fulltop.sh`；
- 结果：`results/gatestack_adaptive_csr_fulltop_20260718/report.{md,json}`。
