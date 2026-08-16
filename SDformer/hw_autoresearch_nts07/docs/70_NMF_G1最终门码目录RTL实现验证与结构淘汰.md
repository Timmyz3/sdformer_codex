# NMF G1最终门码目录RTL实现验证与结构淘汰

**日期**：2026-07-13  
**状态**：首个可综合切片通过本地功能、SVA、lint和Yosys结构检查；未完成真实trace、DC和投影后端  
**对应规格**：`docs/69_HIT-Flow-WG可综合RTL前规格与验证DC准入.md`  
**RTL**：`rtl_hitflow/hitflow_nmf_g1_builder.sv`

## 1. 实现范围

本轮只实现HIT-Flow-WG的保守`G=1`元数据前端，不实现乘积阵列、权重SRAM、分段多播、
accumulator bank或RPI。其功能是把一个窗口内162个token的：

```text
{final Q1.7 gate code, K[31:0]}
```

转换为两类精确事务：

```text
目录term：{group_tag, gate_code, global_input_lane, destination_bitmap[161:0]}
fallback：{group_tag, token_id, gate_code, K[31:0]}
```

目录按最终gate code分配`SLOTS`个slot。同一gate、同一lane只保留一个162-bit目的bitmap；不同
token仍是独立目的位。gate为0或K全零时不生成投影事务。slot溢出时不近似、不丢token，改走
direct fallback。

## 2. 接口与状态机

### 2.1 输入

- `group_valid/group_ready/group_tag`：开始一个独立窗口；
- `token_valid/token_ready`：严格按token 0到161输入；
- `token_gate_code[8:0]`：RTL Shiftmax最终Q1.7码；
- `token_k_bits[31:0]`：当前head的32个K event lane；
- `token_last`：必须且只能在token 161置位。

乱序token、提前last、缺失last或非BUILD状态继续送token都会置`protocol_error`并拒绝事务。

### 2.2 输出

- `term_*`：每拍至多输出一个非空`gate/lane/bitmap`目录项；
- `fallback_*`：单项弹性直通，支持独立ready/valid反压；
- `group_done_*`：目录扫描结束、fallback已退休后才完成；
- `overflow_seen`和四类计数器：token、K-one lane、目录term、fallback token。

### 2.3 状态

```text
IDLE -> BUILD -> DRAIN_DIRECTORY -> [DRAIN_FALLBACK] -> DONE
```

fallback可以在BUILD和目录扫描期间并行退休。若fallback未被接收，builder停止接受下一token；因此
一项弹性存储足以保证无损，代价表现为可测量的上游stall，而不是162深度整帧FIFO面积。

## 3. 存储与周期合同

默认`TOKENS=162、LANES=32、SLOTS=4`：

| 状态 | 位数 | 说明 |
|---|---:|---|
| destination bitmap | 20,736 bit，2.53 KiB | `4×32×162`，寄存器型转置目录 |
| slot gate/valid | 40 bit | 4个9-bit gate加valid |
| fallback elastic | 49 bit加valid/tag | token、gate、32-bit K |
| 控制与计数 | 参数化 | 扫描指针、tag和性能计数器 |

无反压时：

- BUILD接收162个token；
- 目录固定扫描`SLOTS×LANES=128`个lane位置；
- 非空term在对应扫描周期握手，空lane不输出但仍占一拍；
- fallback通常与BUILD/扫描重叠；仅末尾未退休时增加等待；
- 绝对边界延迟还包含group和done握手，论文必须以RTL计数而非只用`162+128`代理。

扫描基址最初用动态`slot×LANES×TOKENS`计算，Yosys保留一个乘法器。最终改为每推进一个lane
将扁平基址加162，去掉动态乘法，generic cell由1469降到1379，mux由500降到409。该数字只用于
结构对比，不是标准单元面积。

## 4. 被工具淘汰的两版结构

### 4.1 动态二维bitmap写口

首版用运行时`destination[slot][lane][token]`更新。功能仿真还发现第二组会读到上一组旧bitmap，
根因是`term_valid`未与`slot_valid`相与。修复代际有效位后，生产参数Yosys仍产生：

```text
destination_q write mux blocks = 663,552
180秒优化超时
```

该写法被淘汰。最终采用编译期展开的128个固定slot/lane bitmap寄存器，每个寄存器只处理整段
初始化和本lane单bit追加；运行时slot选择只作为局部enable。

### 4.2 162深度fallback数组

第二版为最坏情况保存162条fallback。Yosys memory-map后出现4606项未驱动警告，因为未复位但
理论上只读已写地址；即使加复位能消警，也会把约7.8 Kbit变成带复位触发器，并为低概率overflow
支付最坏容量。

最终改为单项弹性fallback：下游阻塞时保持数据并反压输入。生产参数完整memory-map后结构检查
为0问题，且不再有fallback memory。真实trace若显示overflow burst长期连续，才评估深度2/4。

## 5. 验证结果

### 5.1 Icarus功能仿真

通过四组定向场景：

1. 同gate目录合并、第二gate分配、第三gate direct fallback；
2. gate=0和K-zero均不生成projection事务；
3. 连续两个overflow由单项弹性寄存器反压并全部退休；
4. 乱序token报告协议错误且不接收。

同时检查目录bitmap、gate、lane、tag、fallback payload、overflow标志和四类计数器。

### 5.2 Verilator

- 生产参数RTL `--lint-only -Wall`：0 warning、0 error；
- `--binary --assert`：自检testbench和绑定SVA通过；
- SVA覆盖term/fallback/done反压稳定、协议错误拒绝、term非零和DONE时完整token计数。

### 5.3 Yosys

生产参数执行`proc; opt; memory; opt; check; stat`：

```text
Found and reported 0 problems
wire bits     = 134,296
generic cells = 1,379
$dffe         = 132（多数为162-bit宽bitmap寄存器，不能按132个bit理解）
$mux          = 409
$mul          = 0
```

Yosys generic cell不是面积、功耗或频率，不能用于DATE PPA结论。

### 5.4 Erie独立lint

Erie dependency preflight显示远程SSH和FPGA开发依赖缺失，因此只执行本地静态流程。其外部
Verilator后端为0问题；内置Verilog-2001启发式仍报告4个`FOR_CONST_BOUNDS`和4个literal warning。
逐项检查确认对应代码是SystemVerilog的`parameter int`和`genvar/for`常量展开，Yosys生产参数已
成功展开且0结构问题。该结果记录为“工具方言不匹配”，不是Erie全绿；后续若交付纯Verilog-2001
版本，需要单独生成等价wrapper或展开文件，不能把本轮误报静默豁免。

## 6. ASIC质量审阅

### 6.1 已满足

- 单一`clk_core`同步高有效复位域，无CDC/RDC；
- 无组合环、latch、raw gated clock、仿真系统任务或多驱动；
- 所有流接口在反压时保持payload稳定；
- bitmap不复位，但`slot_valid`复位并在新slot分配时初始化全部32个lane，旧代数据不可见；
- fallback、目录和完成具备显式守恒计数；
- 参数化token/lane ID宽度避免`TOKENS=1/LANES=1`形成零宽端口。

### 6.2 未签核风险

| 风险 | 当前等级 | 后续动作 |
|---|---|---|
| 20,736-bit寄存器bitmap面积和时钟负载 | 高 | DC与SRAM/分段表示对照 |
| `SLOTS=4`尚无真实overflow依据 | 高 | ordered final-gate profile后扫描S=2/4/8 |
| 固定128拍目录扫描空lane开销 | 中 | 统计occupied lane；评估slot lane-valid优先编码 |
| 单项fallback可能反压SCS | 中 | 测overflow burst、stall p95/p99；必要时深度2/4 |
| 32-lane popcount为计数器组合链 | 低 | PPA分别报告带/去计数器；必要时离线统计 |
| 尚无真实H67/H68 trace replay | 高 | watcher完成后逐事务比较 |
| 尚无product、weight SRAM和accumulator | 高 | 下一RTL切片实现普通乘法G1后端 |
| 尚无目标库SDC/DC/SAIF/Formality | 高 | 真实参数冻结后进入逻辑综合流程 |

## 7. 对架构创新的影响

本RTL证明的是“最终gate码目录加精确fallback”可以形成可综合控制结构，不证明它已经是DATE创新。
实际设计迭代得到两个有价值的架构约束：

1. 动态目的bitmap必须以固定slot/lane写口表达，否则控制mux会吞噬复用收益；
2. 低概率overflow应优先转换为弹性反压，而不是按最坏情况复制整帧metadata存储。

后续只有在真实trace证明目录term减少、overflow低、固定扫描和bitmap成本可接受，并在完整projection
及full encoder中获得净EDP收益时，NMF才能晋级为论文贡献。当前它是经过结构淘汰的G1公平基线，
不是可直接投稿或直接做芯片签核的完整加速器。
