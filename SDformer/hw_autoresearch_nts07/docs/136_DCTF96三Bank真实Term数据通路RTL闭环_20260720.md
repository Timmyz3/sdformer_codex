# DCTF96 三 Bank 真实 Term 数据通路 RTL 闭环

## 1. 本轮结论

本轮把 DCTF 从“adapter + command fabric 叶模块”推进为包含三套真实 32-lane weight/product/Acc-update 路径的中间顶层：

~~~text
term/event
   |
   v
完整 term 校验与 destination 串化
   |
   v
Q2 三消费者 command fabric
   |             |             |
   v             v             v
bank0 executor   bank1 executor bank2 executor
weight req/rsp   weight req/rsp weight req/rsp
32-lane product  32-lane product 32-lane product
2-way Acc update 2-way Acc update 2-way Acc update
~~~

新顶层为：

~~~text
rtl_hitflow/gatestack_dctf96_term_datapath_top.sv
~~~

当前证据等级为 `[rtl-integration]`。它已经不是纯路由叶模块，但仍未包含三个物理 Acc、bias/final 控制、完整 head/tile 生命周期和真实 S0-S3 trace，因此还不能称为完整 projection accelerator 或 DATE 架构签核。

## 2. 两项 P0 修复

### 2.1 logical supertile 必须逐 command 入队

adapter 发完一个 term 的最后一条 command 后即可接收下一 term，但旧 command 可能仍在 Q2 fabric 或慢 bank 中等待。如果只在顶层保存“当前 logical supertile”，新 term 会覆盖旧值，使旧 command 访问错误的物理权重 tile。

本轮给 fabric entry 增加：

~~~text
cmd_logical_supertile
bank_logical_supertiles[3]
~~~

每条 command 的 supertile 与 tag、sequence、term boundary、channel、gate、lane 和 destination token 一起入队。三个 executor 使用各自 command 携带的值计算：

~~~text
physical_output_tile = 3 * logical_supertile + bank_id
~~~

TB 明确允许第二个不同 supertile 的 term 在前一 term 尚未被全部 bank 完成时进入，并逐 bank 检查物理 tile，没有用“等 fabric 清空再接新 term”掩盖问题。

### 2.2 非法地址必须在 adapter 前拒绝

以下非法 term 不允许“先接受，再只置错误位”：

- `head_input_channel_base + lane_id` 溢出或超过配置的 input channel 数；
- `3 * logical_supertile + 2` 超过物理 output tile 编码范围。

顶层现在使用：

~~~text
adapter_term_valid = term_valid && term_metadata_legal && !flush
term_ready         = adapter_term_ready && term_metadata_legal && !flush
~~~

非法 term 保持不握手、不锁存、不接收 event、不产生 command、weight request 或 Acc update，同时置 sticky `protocol_error`。flush 后合法数据通路可恢复；错误审计位当前保持到 reset。

## 3. 三 Bank 计算语义

每个 `gatestack_dctf32_bank_executor`：

1. 在 term 第一条 destination 上启动一次本地 32-lane weight request；
2. 校验 tag、channel、physical tile 和 epoch；
3. 生成并驻留 32-lane product；
4. 后续 destination 复用同一 product；
5. token 偶数和奇数分别送到两路 Acc update；
6. 最后一条 destination 的 Acc update 被接受后才产生 `term_done`。

因此三 bank 合计暴露：

- 三路独立 32-lane weight req/rsp；
- 六路独立 token-parity Acc update；
- 三路 `bank_term_done`；
- 每 bank completed-term 与 stale-response 计数。

不存在中央 96-lane weight response join、96-lane product join或六路 Acc ready 的强制合并。

## 4. Dispatch 与 Compute 边界

本轮继续区分三种事件：

| 事件 | 含义 |
|---|---|
| adapter issue | 一个 term 的最后一条 destination command 进入 fabric，并建立完成跟踪项 |
| fabric retire | 当前 command 已被三个 bank 各消费一次 |
| bank term done | 对应 bank 的最后一次 Acc update 已握手 |

`head_compute_done` 只在 head-last term 的三个 `bank_term_done` 全部到齐时产生，并检查三路 tag、issue sequence 和 head-last 元数据一致。

需要注意：当前 executor 的 `cmd_ready` 与对应 Acc update 接受绑定，因此最后一条 command 的 fabric retire 和最慢 bank 的 compute complete 可以同拍。本轮已经避免把 retire 单独误报为完成，但尚未实现“命令先进入 bank-local FIFO、计算随后完成”的时间解耦。论文中不能声称已有独立 dispatch/compute pipeline speedup。

## 5. Flush 与 Epoch

flush 当拍组合屏蔽：

- term/event ready；
- weight request 和 response ready；
- 六路 Acc update；
- 三路 bank term done；
- fabric retire；
- head compute done。

下一拍清空 adapter、fabric、executor 活动状态和 term completion tracker。三个 executor 的 epoch 各自递增，旧 epoch weight response只被 drain/drop并计数，不进入 product、Acc 或完成路径。

默认 `EPOCH_W=4` 仍有有限回绕约束：系统必须保证旧响应在16次 epoch 递增前排空，或增大 epoch/增加未决请求跟踪。

## 6. 独立复跑结果

### 6.1 DCTF fabric

| Q | cycles | accepted | retired | max occupancy |
|---:|---:|---:|---:|---:|
| 2 | 402 | 260 | 256 | 2 |
| 3 | 391 | 260 | 254 | 3 |
| 4 | 387 | 260 | 252 | 4 |

测试包含随机 bank 反压、两次随机 flush、full retire+accept 同拍复用、逐 bank command 顺序与 supertile sideband 比较。accepted 与 retired 的差异来自有意 flush 的在途 command，不是丢失。

### 6.2 DCTF96 中间顶层

| 指标 | Icarus | Verilator |
|---|---:|---:|
| 总周期 | 91 | 89 |
| issued terms | 4 | 4 |
| completed terms / bank | 3/3/3 | 3/3/3 |
| weight requests / bank | 4/4/4 | 4/4/4 |
| Acc updates / bank | 6/6/6 | 6/6/6 |
| 六路 parity updates | 每路3 | 每路3 |
| stale responses / bank | 1/1/1 | 1/1/1 |
| mismatch | 0 | 0 |

四个 issued term 包括两个正常 term、一个随后被 flush 取消的 term 和一个相同 SRAM 身份的新 epoch 替代 term，因此每 bank只完成三个 term。

### 6.3 工具链

| 检查 | 结果 |
|---|---|
| Icarus 自检仿真 | PASS |
| Verilator 动态 SVA | PASS |
| Yosys hierarchy/check/stat | PASS |
| Erie RTL/TB lint | 0 error / 0 warning |

Yosys hierarchy 总计为1473个 generic cells、0个 process，其中包含三套 32-lane product engine；该数字不是目标库面积。日志中的 memory-to-register 替换提示被显式分类为预期结构提示。

## 7. 当前能主张什么

可以主张：

1. DCTF 已实现为三 bank 独立权重、乘积驻留和六路 Acc-update 的真实 RTL 数据通路；
2. logical supertile 随 command 驻留，支持跨 term 重叠而不串物理 tile；
3. 完成跟踪严格绑定最后一次 Acc update，而不是只看 command retire；
4. flush + epoch 对同 tag/channel/tile 的迟到 weight response 保持零污染。

仍不能主张：

- DCTF 相对 Central96 或 Independent96 的真实性能、能耗或 EDP提升；
- 完整 Acc/bias/final 已实现；
- dispatch 与 compute 已在时间上完全解耦；
- 真实 H67 S0-S3 全元素 bit-exact；
- DC、STA、SAIF、SRAM macro或布局布线签核。

## 8. 下一阶段 P0

1. 给 `hitflow_banked_accumulator` 增加显式同步 flush，清除 group active、bank busy、valid bitmap和旧 final，不清性能累计计数；
2. 接入三个 `OUT_TILE=32, BANKS=2` Acc，形成六路相同物理口径；
3. 使用三条独立 32-lane 同步 bias 通道，响应校验 tag/tile/token/epoch；
4. 三 Acc group start/finish原子握手，六路 final 独立反压；
5. 明确 overflow 后的整 tile 原子提交策略；
6. 使用同一 H67 S0-S3 trace比较 Central96、3xIndependent32和DCTF96。

只有完成上述六项并达到文档131的公平门槛，DCTF才能从 `[rtl-integration]` 晋级为论文主架构候选。
