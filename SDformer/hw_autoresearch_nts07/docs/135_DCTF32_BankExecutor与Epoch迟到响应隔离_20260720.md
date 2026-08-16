# DCTF32 Bank Executor 与 Epoch 迟到响应隔离收口

## 1. 本轮边界

本轮不修改RTL、TB或SVA，只为现有`gatestack_dctf32_bank_executor.sv`补充同源开放库映射代理和证据收口。比较对象为：

1. `gatestack_decoupled_product_engine`叶模块；
2. 包含该叶模块的`gatestack_dctf32_bank_executor`。

两次映射读取完全相同的两份RTL：

~~~text
rtl_hitflow/gatestack_decoupled_product_engine.sv
rtl_hitflow/gatestack_dctf32_bank_executor.sv
~~~

两者使用相同Yosys流程、相同`NangateOpenCellLibrary_typical.lib`，并把内部乘积引擎对齐为`OUT_TILE=32、TAG_W=36`。其中36位tag对应executor默认的`GROUP_TAG_W=32`与`EPOCH_W=4`拼接。

## 2. 两个顶层的功能边界不同

Product engine叶模块负责：

- 锁存一条term的tag、gate、channel、tile和issue sequence；
- 发出一次权重请求并校验响应身份；
- 计算并寄存32个lane的乘积；
- 通过ready/valid保持乘积输出；
- 暴露五组term、请求、乘积、等待和停顿计数器。

Bank executor在此基础上还负责：

- 校验command起始、连续sequence、term first/last和head-last协议；
- 将logical supertile映射到本bank物理tile；
- 为请求附加epoch并drain/drop旧epoch响应；
- 在整条term期间驻留同一份product，供多个destination复用；
- 按token奇偶选择两路Acc update端口并处理反压；
- 在最后一条destination更新接受时产生term完成。

因此executor映射包含协议、乘积驻留和路由开销，但两个顶层不是同一功能边界。尤其叶模块顶层的五组计数器是可观察输出，而嵌入executor后对应内部计数器未使用并可被优化。最终差值是不同可观察边界下的净差值，不是wrapper毛开销，更不是纯路由面积或物理互连面积。

## 3. 已有RTL验证事实

现有`sim_hitflow/run_gatestack_dctf32_bank_executor_checks.sh`和对应日志记录：

| 检查 | 结果 |
|---|---:|
| Icarus | PASS |
| Verilator动态SVA | PASS |
| Yosys | PASS |
| Erie RTL | PASS，0 error / 0 warning |
| Erie TB | PASS，0 error / 0 warning |

测试覆盖多destination、单destination、奇偶Acc路由、权重请求反压、Acc反压、错误响应身份、错误command元数据、flush和旧epoch响应。

### 3.1 ABA迟到响应用例

ABA用例执行顺序为：

1. 发出旧epoch权重请求；
2. 在响应返回前flush；
3. 使用相同tag、input channel和physical tile发出新epoch请求；
4. 旧epoch响应先到，仅凭epoch不匹配被ready/drop；
5. drop拍不产生Acc update或`term_done`；
6. 新epoch响应随后到达并正常完成。

最终动态结果为：

~~~text
PASS DCTF32 BANK EXECUTOR requests=2 updates=1 done=1 stale_rsp=1
~~~

这证明当前测试窗口内，旧epoch响应先到时不会污染替代请求，且审计计数`stale_rsp=1`。

## 4. Epoch有限回绕约束

RTL默认`EPOCH_W=4`，epoch只有`2^4=16`个状态。当前判定依据是响应epoch是否等于当前`epoch_q`，因此它不是无限生命周期的请求标识。

系统正确性必须保证：旧响应在epoch计数绕回同一值之前全部排空。若同一旧响应能跨越16次epoch递增后返回，它会与新请求发生ABA别名，仅靠当前4位epoch无法区分。

集成约束至少选择一项：

- 限制SRAM最大响应寿命，并保证该窗口内flush次数小于16；
- 在允许下一次复用前确认所有旧请求已排空；
- 扩大`EPOCH_W`；
- 增加未决请求表或更强的generation标识。

现有ABA测试证明一次flush后的旧响应隔离，不证明任意次数flush后的无限期隔离。

## 5. Nangate45映射结果

证据等级仅为**开放库无约束logic proxy**。

| 顶层 | 库面积值 | 标准单元数 | `$mem_v2` |
|---|---:|---:|---:|
| 32-lane product-engine叶模块，36-bit tag | 20040.706 | 15409 | 0 |
| DCTF32 bank executor | 20367.886 | 15643 | 0 |
| executor净增 | 327.180 | 234 | 0 |

Executor相对叶模块的库面积值增加`1.633%`，标准单元数增加`1.519%`。该净差值同时混合command协议、epoch隔离、乘积驻留、奇偶Acc路由、完成控制和可观察输出优化，不能拆解或命名为纯路由面积。

### 5.1 映射质量核查

- 两个顶层的`check -assert`均报告0问题；
- 最终统计均为0个memory、0个process，`$mem_v2=0`；
- 最终网表非空，且未发现`$`前缀未映射单元；
- 两份日志各有8类、共72条Liberty解析提示，均为扫描触发器`SE*SI+D*!SE`函数表达式不受支持；最终映射统计使用`DFF_X1`和目标库组合单元，不含这些扫描触发器。

输入文件与Liberty的SHA-256记录在`results/gatestack_dctf32_bank_executor_20260720/input_sha256.txt`，Yosys版本记录在同目录`yosys_version.txt`。

## 6. 证据限制

本结果没有：

- SDC或时钟、输入输出延迟约束；
- STA、WNS、TNS或关键路径；
- SAIF或任何活动率功耗分析；
- SRAM macro、memory compiler或存储外围；
- DC、Formality或商业目标库结果；
- 布局布线、时钟树和真实互连寄生。

因此`20040.706`、`20367.886`和`327.180`都只是同源逻辑结构筛选数字，不是面积平方微米结论，不得称为ASIC PPA或签核结果。

## 7. 复现

~~~bash
python3 scripts/test_summarize_gatestack_dctf32_bank_executor.py
bash dc_handoff/scripts/run_gatestack_dctf32_bank_executor_nangate45_mapping.sh
~~~

结构化结果、中文报告、映射日志、映射网表、输入哈希和工具版本位于`results/gatestack_dctf32_bank_executor_20260720/`。
