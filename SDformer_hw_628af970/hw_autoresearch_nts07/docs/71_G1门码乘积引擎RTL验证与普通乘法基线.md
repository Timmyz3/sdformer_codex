# G1门码乘积引擎RTL验证与普通乘法基线

## 1. 本轮回答的问题

NMF G1目录已经能够把同一窗口内相同最终Q1.7 gate码、相同K输入通道的目的token合并成
bitmap。本轮继续实现目录之后的普通乘法基线，回答三个问题：

1. 最终gate码能否按无符号正数与有符号int8权重精确相乘；
2. 权重请求、响应和乘积输出能否在反压下保持事务与数据稳定；
3. 后续CSD、共享乘法器或其他电路优化应当与什么可综合基线比较。

本轮不包含目的token多播累加、bias提交、输出量化、BN折叠验证、完整C乘C projection、
真实SRAM宏、DC或SAIF。因此它是G1投影后端的第二个RTL切片，不是完整加速器签核。

## 2. 精确数值合同

输入gate码为9-bit无符号数，部署有效范围为1到256；gate=0由NMF前端过滤。权重为signed
int8，范围为-128到127。乘积范围为：

```text
最小值 = 256 * (-128) = -32768
最大值 = 256 * 127    = 32512
```

因此使用signed 17-bit乘积，范围为-65536到65535，具有充分余量。RTL先把gate零扩展为
10-bit signed正数，再与signed int8权重相乘，并显式转换到`PRODUCT_W=17`，不依赖工具的
隐式截断规则。

Python参考测试穷举gate `0..256`与int8权重`-128..127`的全部65,792种组合，验证全部乘积
落在signed 17-bit范围，并验证17位二补码编码和解码逐项一致。

## 3. 模块与接口

生产RTL为`rtl_hitflow/hitflow_gate_product_engine.sv`，默认参数为8个输出lane。模块采用单时钟、
同步高有效复位和ready/valid协议，状态机为：

```text
IDLE
  -> WEIGHT_REQUEST：保持{tag,input_channel,output_tile}
  -> WEIGHT_RESPONSE：只接受tag、输入通道和输出tile全部匹配的响应
  -> PRODUCT_OUTPUT：保持8路signed 17-bit乘积和目的bitmap
  -> IDLE
```

输入复用键在系统层必须是：

```text
{block_id, final_gate_code, global_input_channel}
```

NMF当前输出head内局部lane；集成层必须计算`global_input_channel=head_id*32+lane_id`，并保证
不同block不共享权重响应。opaque tag用于携带窗口组、block、head或其他调度身份，当前叶模块
不解释tag字段。

模块统计接收term数、权重请求数、乘积提交数、等待权重响应周期和输出阻塞周期。它没有统计
权重请求端阻塞周期；该口径在testbench中已按定义修正。

## 4. 定向验证

### 4.1 Icarus

定向场景包括：

1. 权重请求连续反压时请求tag、输入通道和输出tile稳定；
2. 错tag权重响应必须拒绝并报告协议错误；
3. `256*(-128)`、`256*127`和`256*(-1)`边界精确；
4. `64*2`、`64*(-3)`和`64*0`普通正负乘积精确；
5. 乘积输出反压时valid、bitmap和数据保持；
6. 零gate或空目的bitmap不得被接收；
7. 事务、权重请求、产品、响应等待和输出阻塞计数器符合定义。

结果为全部通过。Icarus关于`unique case`和`always_comb`常量选择的`sorry`信息是该模拟器的
能力提示，不是RTL错误；相同设计继续由Verilator和Yosys独立交叉检查。

### 4.2 Verilator与SVA

- 生产参数RTL执行`--lint-only -Wall`：0 warning、0 error；
- 带绑定SVA的二进制仿真通过；
- SVA覆盖权重请求反压稳定、乘积反压稳定、非空目的bitmap和非法term拒绝。

testbench最初使用32-bit integer读取17-bit乘积，产生宽度扩展告警。本轮改为同宽signed
17-bit比较，并实际检查返回的输入通道、输出tile和两类stall计数，最终验证构建日志无告警。

### 4.3 Yosys

默认`OUT_TILE=8`执行`proc; opt; memory; opt; check; stat`：

```text
Found and reported 0 problems
wire bits = 1,754
cells     = 77
$mul      = 8
memory    = 0
```

这说明结构与“每个输出lane一个普通乘法器”的基线一致。generic cell数量不是标准单元面积，
`$mul=8`也不能直接推出目标工艺中的乘法器实现、时序或功耗。

### 4.4 Erie独立静态检查

Erie内置启发式对`genvar for`初始化报告1项`LITERAL_BASE_WIDTH` warning；对应行是标准
SystemVerilog常量生成循环。生产RTL已通过Verilator完整展开和Yosys默认参数结构检查，因此记录为
工具方言提示，不伪报为Erie全绿，也不为迎合启发式改写正常的参数化生成结构。

## 5. 统一可复现回归

新增：

```bash
hw_autoresearch_nts07/sim_hitflow/run_projection_g1_checks.sh
```

该入口串行执行NMF和普通乘法引擎的Icarus、生产RTL Verilator lint、绑定SVA、默认参数Yosys以及
投影代数和位宽Python测试。本轮执行结果全部通过：NMF为1,379个Yosys generic cells且无乘法，
普通乘法引擎为77个generic cells和8个乘法器，二者均为0个Yosys结构问题。

## 6. 对架构设计的约束

1. 普通乘法器是后续CSD、移位加法或跨term共享优化的公平D1基线，任何优化必须在相同
   OUT_TILE、频率、SRAM响应和真实trace下比较；
2. 当前四状态单发射实现强调接口正确性，不代表最终吞吐。若真实trace显示产品后端成为瓶颈，
   应增加多entry outstanding或流水乘法，而不是先假设乘法器主导；
3. 产品只生成一次不等于累加只做一次。每个目的token仍必须对各输出lane执行独立累加，后续
   分段多播和accumulator bank冲突可能吞掉乘法复用收益；
4. 真实收益必须使用最终RTL gate码统计。score class、浮点gate或跨block相同数值均不能作为复用；
5. 蝶形网络仍是条件候选。只有简单分段多播出现可测inter-segment瓶颈，且蝶形版本完整projection
   EDP改善达到文档69门槛时才实现。

## 7. 下一步

下一RTL切片是带目的bitmap的分段多播累加器，至少实现：

- 每个token、每个output tile独立的signed宽位累加状态；
- bitmap展开、bank映射和写冲突处理；
- product反压与目的提交守恒；
- 每个token/output tile一次bias提交；
- direct fallback与NMF term进入同一乘积和累加后端；
- 普通direct、NMF G1和后续G2/G4共享同一接口与计数口径。

真实H67/H68 ordered profile仍在等待软件训练队列释放A800。profile完成前可以实现G1正确性基线，
但不能冻结slot数、窗口组大小、分段宽度、蝶形互连或论文主贡献。
