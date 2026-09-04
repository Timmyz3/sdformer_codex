# PPDI 双目的 Bank Executor 叶模块与 Exactly-Once 提交

## 1. 本轮边界

本轮只实现 PPDI 的单 bank executor，不修改标量 DCTF 基线，也未连接 PPDI adapter、fabric 或完整 projection。目的不是提前宣称 H67 加速，而是验证最危险的状态语义：一个 term product 能否安全服务一偶一奇两个目的 token。

新增文件：

- `rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv`；
- `tb_hitflow/tb_gatestack_ppdi_dctf32_bank_executor.sv`；
- `verif_hitflow/gatestack_ppdi_dctf32_bank_executor_assertions.sv`；
- `verif_hitflow/bind_gatestack_ppdi_dctf32_bank_executor_assertions.sv`；
- `sim_hitflow/run_gatestack_ppdi_dctf32_bank_executor_checks.sh`。

## 2. 接口变化

标量 executor 的单一 `cmd_destination_token` 被替换为：

| 信号 | 宽度 | 语义 |
|---|---:|---|
| `cmd_destination_valid` | 2 | bit0 为偶 token，bit1 为奇 token；至少一位有效 |
| `cmd_destination_tokens` | `2*TOKEN_ID_W` | port0/port1 对应偶/奇 token |
| `acc_update_valid/ready` | 2 | 两个 Acc parity port 独立 ready/valid |
| `destination_done_q` | 2 | 当前 command 已成功提交的目的掩码 |

其余 group tag、term issue sequence、gate、input channel、lane、supertile、weight request/response、term-done 和 epoch 接口保持与标量 executor 同构。

## 3. 数据流与状态

~~~text
dual-destination command
        |
        +--> 首command启动一次product engine --> 一次weight request
                                            |
                                      product resident
                                            |
                         +------------------+------------------+
                         |                                     |
                  even Acc port                         odd Acc port
                         |                                     |
                    done_mask[0]                          done_mask[1]
                         +------------------+------------------+
                                            |
                               所有有效位完成后cmd_ready
                                            |
                              term-last时释放product/done
~~~

组合式 valid 不依赖 ready。`acc_update_valid` 只由有效目的和 `~destination_done_q` 决定；某端口握手后，对应 done 位锁存，后续周期不再向该端口发射。`cmd_ready` 只有在所有有效目的已经提交或将在当前拍提交时才拉高。

这形成两层边界：

1. Acc port 是独立 exactly-once commit，可分拍完成；
2. fabric command 是整体 retire，只有全部有效目的提交后才能前进。

原候选规格要求两个端口同拍握手。RTL 将其改为 commit-mask 方案，因为强制同拍要么让 valid 依赖 ready，要么需要上游保证两个 ready 同时出现。commit mask 增加 2 bit 状态，但支持独立反压且不重复累加。

## 4. 合法性合同

命令必须满足：

1. destination valid mask 非零；
2. port0 有效时 token 必须为偶数且小于 `TOKENS`；
3. port1 有效时 token 必须为奇数且小于 `TOKENS`；
4. first/continuation、sequence、tag、issue sequence、channel、gate、lane 和 supertile 与标量 DCTF 规则一致；
5. head-last 只能出现在 term-last command；
6. source 在活动命令未 retire 前保持所有 command 字段稳定。

非法命令置 sticky `protocol_error`，不会启动 weight 或 Acc 更新。clear 与新错误同拍采用 new-error-wins。

## 5. Flush 与迟到响应

若偶端口已写、奇端口尚未写时发生 flush：

1. executor 当拍屏蔽 command、weight response、Acc 和 term-done；
2. 清除 term-active 与 done mask；
3. 有未返回请求时把其 epoch 放入 pending-generation bitmap，只从空闲 generation 中分配新 epoch；
4. pending generation 全满时 fail-closed，阻止新 term，直到一个旧 response 被 drain 后恢复；
5. 旧 weight response 只做 ready/drop 并计数；
6. 在完整 projection 中，同一 flush 还会清 Acc group valid bitmap，因此部分写入不可成为后续有效 final。

叶模块本身没有回滚外部 SRAM 写入。本轮已把 executor 接到真实 `hitflow_banked_accumulator`，用同一 flush 验证旧偶端口部分写在同 tag/token 重启后不进入 final；完整三 bank projection 仍需再次验证该合同。

## 6. 验证结果

结果包：`results/gatestack_ppdi_dctf32_bank_executor_20260722`。

| 运行 | 非reset周期 | command | weight | Acc偶/奇 | done | stale | mismatch |
|---|---:|---:|---:|---|---:|---:|---:|
| Icarus | 177 | 7 | 16 | 6/6 | 5 | 3 | 0 |
| Verilator 动态SVA | 177 | 7 | 16 | 6/6 | 5 | 3 | 0 |

定向覆盖包括：

- 偶端口先提交、奇端口延迟提交，偶端口不重复；
- 偶奇同拍提交；
- only-even、only-odd；
- 两个 command 复用一次 term product；
- 空 mask 与奇偶编码错误；
- 部分提交时 flush；
- 同身份旧 epoch weight response 丢弃。
- Acc valid 受阻期间并行 drain pending stale，valid 不撤回；
- child 旧 sticky error 单拍 clear 后 parent/child 同时清除；
- paired non-last 到 paired last 只释放一次 product，且覆盖奇端口先提交；
- `EPOCH_W=3` 下填满 8 个 pending generation 时阻塞，drain 后恢复而不发生 ABA 复用；
- 每个 pending generation 保存 tag/channel/tile，错误身份 response 不清位，完整身份匹配后才释放；
- 真实双 bank Acc 中旧 partial write 经共同 flush 后，在同 tag/token 重启 final 中不可见。

真实 Acc 集成的两个模拟器结果均为：旧 partial commit 偶/奇=`1/0`，恢复后 token2 final=`1`，bias=`4`，updates/writes=`6/6`，逐 lane mismatch=0。Yosys hierarchy/check/stat 通过；Erie RTL/TB 为 0 error、0 warning。输入 SHA256、工具版本、可搬移日志及日志 SHA256 均在结果包中固化。

## 7. 当前可写与不可写

可写：PPDI executor 通过 exactly-once commit mask 把 term 内 product multicast 映射到已有偶/奇 Acc 双端口，并保持命令整体 retire。

不可写：

- PPDI 已减少 H67 周期 30.270%；
- PPDI 完整 projection 已 bit-exact；
- PPDI 已改善面积、功耗或 EDP；
- 本轮动态 SVA 等于 formal 签核。

下一步必须先独立审阅本叶模块，再实现 whole-term parity partition adapter 与双目的有序 fabric。
