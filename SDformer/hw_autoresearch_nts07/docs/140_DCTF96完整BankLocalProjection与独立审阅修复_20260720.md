# DCTF96完整Bank-Local Projection与独立审阅修复

## 1. 本轮推进

此前DCTF96只到`term/event -> 三bank weight/product -> 六路Acc update`，还不是完整projection。本轮新增完整顶层，将三套真实`hitflow_banked_accumulator`、三路同步bias和六路final接入，并补齐tile/head生命周期、flush恢复和错误完成合同。

~~~text
tile/head controller
       |
       v
term/event adapter -> Q2 multi-reader command fabric
                          |       |       |
                       bank0   bank1   bank2
                          |       |       |
                      weight/product executor x3
                          |       |       |
                      2-bank Acc x3
                          |       |       |
                      bias SRAM port x3
                          |       |       |
                       final x2 x3
~~~

共享边界只携带窄命令。权重、product、Acc读改写、bias和final都保持bank-local，没有中央768-bit weight join或96-lane product总线。

## 2. 生命周期状态机

顶层使用七态控制：

~~~text
IDLE -> WAIT_HEAD -> RUN_HEAD -> HEAD_DONE
          ^                           |
          +---------------------------+  非末head
                                      |
                                      v  末head
BIAS -> FINISH -> TILE_DONE -> IDLE
~~~

- tile start同时启动三套Acc group，三者必须原子握手；
- head metadata按`tag/index/last`顺序校验；
- `source_done`可以提前到达并锁存，但必须等待DCTF真实排空；
- 三个bias port各自单outstanding，可独立错峰；
- 三套Acc都完成162次bias commit且busy清空后，group finish才原子提交；
- final是六个物理ready/valid通道，任一路反压不要求其他路同步停顿。

## 3. Bias与恢复合同

每个projection bank使用独立bias request/response身份：

~~~text
{tag, physical_output_tile, token, epoch}
~~~

物理输出tile固定为`3 * logical_supertile + bank_id`。正常current-epoch响应只有身份完全匹配才可形成Acc bias commit；wrong-current响应被吞掉并置错，但保持原outstanding，防止ready/valid源永久堵塞；旧epoch响应直接drop并按bank计数。

外部flush可以保持多个周期，但内部DCTF epoch只接收首拍pulse，bias epoch也只在首拍递增。三套Acc仍接收原始flush电平，用于整个flush期间持续屏蔽接口和清除group-local状态。

## 4. 独立审阅发现与关闭

| 级别 | 问题 | 修复 |
|---|---|---|
| P0 | `source_done`与新term同拍时，拍前idle可能提前完成head | 完成条件加入“该拍无term/event握手” |
| P1 | wrong-current bias响应`ready=0`会让合规响应源永久保持valid | 原子吞掉、置错、不commit、不清outstanding |
| P1 | 长flush每拍推进4-bit epoch，可能16拍回绕 | flush首拍检测，只推进一次 |
| P1 | overflow后仍可接新tile，后续tile永久误报 | overflow阻止新tile，flush后恢复 |
| P1 | done error由组合sticky生成，反压期间可变化 | 进入HEAD_DONE/TILE_DONE时锁存 |

修复后重新运行动态SVA，所有完成payload在反压期间稳定。

## 5. 验证证据

专用入口：

~~~bash
bash sim_hitflow/run_gatestack_dctf96_banklocal_projection_checks.sh
~~~

结果为：

- Icarus：PASS；
- Verilator动态SVA：PASS且构建0 warning/error；
- Yosys hierarchy/check/stat：PASS，0 process；
- Erie RTL与TB：均为0 error、0 warning；
- 定向统计：4个head、2个term、18个final逐项检查；
- 覆盖双head、多destination、zero-term、三bank错峰、六final反压、并发done、wrong-current、长flush、旧epoch和同tag恢复。

机器可读与中文结果位于`results/gatestack_dctf96_banklocal_projection_20260720/`。

## 6. 开放库映射代理

默认`Q=2/TOKENS=162/OUT_TILE=32`完整flatten后，Nangate45无约束映射结果为：

| 逻辑库面积值 | cell | `$mem_v2` | process | 映射网表字节 |
|---:|---:|---:|---:|---:|
| 182719.124 | 135045 | 11 | 0 | 19566055 |

除`$mem_v2`外没有未映射`$`单元。该结果只证明完整层级可形成非空映射网表，memory面积未计，也没有时序或功耗约束。中文报告位于`results/gatestack_dctf96_banklocal_projection_mapping_20260720/report.md`。

## 7. 架构意义与边界

本轮把DCTF从“多消费者FIFO + term executor”推进为可运行的完整projection后端，已经可以回答bank-local累加、bias和final是否能在统一生命周期下工作。它仍不能单独证明DATE级架构收益，因为缺少三个公平对照的真实trace周期与同工艺能量。

当前不能宣称：

- DCTF相对Central96或Independent32已经加速或降低EDP；
- Yosys generic cell是目标面积；
- `$mem_v2`数量等于SRAM面积；
- 已完成DC、STA、SAIF、LEC或P&R；
- 已达到DATE accept标准。

## 8. 下一步

1. 用H67 S0-S3真实term/token、INT8 weight、acc32 bias和expected输出回放DCTF96；
2. 建立相同term/event输入边界的Central96与3xIndependent32后端，不混入decoder复制差异；
3. 单独再比较完整decoder边界，量化decode-once与三读slot的代价；
4. 在同一三weight bank、六Acc bank、bias latency和final sink下报告wall time、bank skew、物理访问和逐元素mismatch；
5. 只有达到文档131的EDP或面积/能量门槛，才将DCTF升级为主架构贡献。
