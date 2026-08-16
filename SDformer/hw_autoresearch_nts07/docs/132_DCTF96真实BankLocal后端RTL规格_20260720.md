# DCTF96 真实 Bank-Local 后端 RTL 规格

## 1. 当前叶模块为什么还不是架构

现有 `gatestack_dctf_term_fabric` 已实现三消费者有序 command queue，但它只证明控制命令可以被三个 bank 各消费一次。它还不能直接接真实 projection，原因有两个：

1. entry 只有 command sequence 和一个 destination token，缺少 `term_issue_seq/term_first/term_last/head_last`；真实 decoder 是“一条 term 对应一个或多个 event beat”；
2. `retire_valid` 只表示三个 bank 都接受了该 destination command，不表示 weight response、product 和 Acc 写回已经完成。

因此必须区分：

~~~text
dispatch retire != bank compute complete != head complete
~~~

在这三个阶段未分离之前，DCTF 只能记作 `[rtl-leaf]`，不能列为 DATE 架构贡献。

## 2. 目标结构

新增 sibling top，保留 Central 路径不改：

~~~text
shared resident/IPD/FADC/RAW decoder
                 |
                 v
      term/event exact adapter
                 |
                 v
       DCTF Q=2 command fabric
          /          |          \
  32-lane bank0  bank1      bank2
  weight SRAM    weight SRAM weight SRAM
  product cache  product     product
  2 Acc banks    2 Acc banks 2 Acc banks
  bias/final     bias/final  bias/final
~~~

建议模块边界：

- `gatestack_dctf_term_event_adapter`：将 term + 4-way event 串化为 exact destination command；
- `gatestack_dctf_term_fabric`：Q=2 三消费者 command queue；
- `gatestack_dctf32_bank_backend`：单个32-lane weight/product/Acc路径；
- `gatestack_dctf96_multihead_tile_projection_top`：三 bank 生命周期、bias/final、完成和错误汇聚。

共享 decoder 之外，不再生成中央 768-bit weight、1632-bit product 或 96-lane final join。

## 3. Adapter 合同

### 3.1 输入

保持当前 backend 的 term/event 流：

- term：`tag, gate_code, lane_id, destination_count, issue_seq, head_last`；
- event：`gate_code, lane_id, token_valid[3:0], token_ids[3:0], event_count, issue_seq, term_first, term_last, head_last`。

### 3.2 输出 command

每个有效 destination 产生一条窄命令：

~~~text
{group_tag,
 cmd_sequence,
 term_issue_seq,
 input_channel,
 logical_supertile,
 gate_code,
 lane_id,
 destination_token,
 term_first,
 term_last,
 head_last}
~~~

其中：

- `cmd_sequence` 对每个 destination 单调递增，用于 fabric 三消费者保序；
- `term_issue_seq` 在同一 term 的所有 destination 上保持不变；
- `logical_supertile` 必须随每条 command 写入 fabric entry，不能只保存在 adapter 顶层寄存器；adapter 发完一个 term 后可以接收下一 term，而旧 command 此时可能仍在 Q2 或 bank 端等待；
- `term_first` 只在该 term 第一条 destination command 为 1；
- `term_last` 只在最后一条为 1；
- `head_last` 只允许与最后一个 term 的 `term_last` 同时出现。

### 3.3 exact 检查

adapter 维护一份中央 162-bit seen bitmap，仅用于范围、重复和计数验证，不向 bank 广播 bitmap。必须检查：

1. token `< TOKENS`；
2. 同一 term 内 token 不重复；
3. event gate/lane/issue_seq 与当前 term 一致；
4. first/last 顺序合法；
5. event_count 等于 token-valid popcount；
6. term 结束时实际 destination 数等于 descriptor count；
7. `head_input_channel_base + lane_id` 不得溢出 `INPUT_CH_W`；
8. 错误 term 不得产生部分、重复或越界 command。

## 4. 单 Bank 状态机

每个 `gatestack_dctf32_bank_backend` 独立运行：

| 状态 | 行为 |
|---|---|
| B_IDLE | 等待 `term_first`，锁存 term identity 和首 token |
| B_WEIGHT_REQ | 向本 bank 32-lane SRAM 发一次权重请求 |
| B_WEIGHT_WAIT | 等待并校验 tag/input-channel/physical-tile/epoch |
| B_APPLY_FIRST | product valid 后向首 token 的 Acc bank 更新 |
| B_RUN_TERM | 后续 command 复用本地 product，逐 token 更新 Acc |
| B_TERM_DONE | 最后一条 update 接受后释放 product并发 term-complete |

物理地址：

~~~text
physical_output_tile = 3 * logical_supertile + bank_id
~~~

现有 `gatestack_decoupled_product_engine(OUT_TILE=32)` 可以复用。其 product output 寄存器在 `product_ready=0` 时保持稳定，因此可作为整条 term 的本地 product cache；只有最后一条 destination 的 Acc update 同拍接受时才拉高 `product_ready`。

## 5. Q=2 Fabric 与 Bank Skew

三个 bank 有独立 `ready`，快 bank 可以在队列允许范围内领先慢 bank。每条 command 的三位 consume mask 保证各 bank 恰好消费一次，bank 内顺序与输入一致。

fabric entry 必须同时存储 `logical_supertile`。这是跨 term 正确性要求，而非性能选项：仅在顶层锁存当前 supertile 会在“新 term 已接收、旧 term command 尚未被慢 bank 消费”时产生物理权重 tile 串扰。

必须新增并报告：

- `bank_pending`、`occupancy`、input stall；
- 每 bank weight wait、product stall、Acc stall；
- bank completion skew 的 p50/p95/p99/max；
- dispatch-retired、term-completed 和 head-completed 三类计数。

Q=2 是默认起点。现有 synthetic Q4 相对 Q2 只改善 3.23% 且面积增加 33.9%，不能直接外推到真实 term-level 数据流。

## 6. Head 完成条件

正确的 head done 条件为：

~~~text
source_done_seen
&& adapter_empty
&& fabric_occupancy == 0
&& completed_terms[0] == issued_terms
&& completed_terms[1] == issued_terms
&& completed_terms[2] == issued_terms
~~~

fabric 的 `retire_valid` 不能单独触发 head done。最后一个 term 在三个 bank 的最后一条 Acc update 都接受之后，才允许进入 bias/final。

## 7. Abort、迟到响应与 Final

任一 decoder、adapter、fabric、weight、product、Acc 或 overflow 错误：

1. 立即禁止新 command 和 final；
2. 原子 flush DCTF 和三个 bank；
3. 上层只产生一次 tagged error completion；
4. reset/flush 后不得有 stale final、重复 completion 或旧 command。

若 weight SRAM response 不能取消，请求/响应必须带 epoch。旧 epoch response 只 drain/drop，不能进入 product engine。

最终接口暴露六路 32-lane final channel，每个 bank-local accumulator 两路 token-interleaved final。`tile_done` 仅在三个 local group-finish 全部完成后产生；不能为接口好看重新拼成原子 96-lane 宽总线。

## 8. 复用与禁用

直接复用：

- resident/IPD/FADC/RAW decoder；
- replay mux；
- 三个 `gatestack_decoupled_product_engine(OUT_TILE=32)`；
- 三个 `hitflow_banked_accumulator(OUT_TILE=32, BANKS=2)`；
- context abort controller。

DCTF 路径禁用：

- `gatestack_term_fork`；
- `gatestack_product_bitmap_join`；
- `hitflow_segmented_multicast`；
- `gatestack_hatf96_weight_coalescer`；
- 任何中央 weight/product/final 宽 join。

这些模块继续保留给 HATF96-Central 公平对照。

## 9. 验证矩阵

### 9.1 Adapter/Fabric 叶模块

- 单 destination、多 event beat、4-way 满载；
- token 重复、越界、event count 错、gate/lane/seq 错；
- term first/last/head-last 边界；
- bank 随机反压、queue full、retire+accept 同拍、flush；
- 每 bank command payload/顺序逐项一致。

### 9.2 Bank Backend

- 三种 weight response 顺序和不同延迟；
- 错 tag/channel/tile/epoch；
- 单 term 多 destination product 只请求一次；
- token%2 Acc bank冲突与反压；
- 最后一拍 abort和迟到 response；
- product/Acc 数值逐 lane exact。

### 9.3 真实 S0-S3

- 与 Central 和 3xIndependent32 使用同一 payload、weight、bias、expected acc32；
- 四阶段物理 weight access 保持 `15030`，payload `3847680 bit`；
- 六路 final 全元素比较；
- 同步 bias、final随机反压、错误原子清理；
- 报告 wall cycle、slot read、decoder term、bank skew 和面积。

## 10. 晋级门槛

DCTF 只有在真实 bank-local top 完成后才可进入主贡献候选。相对 `3xIndependent32` 必须满足：

- projection EDP 改善至少 `15%`；
- 或总能量与总面积各改善至少 `10%`；
- 同时相对 HATF96-Central 不出现明显周期退化，并改善宽网、时序或能量证据。

如果只完成 adapter+fabric 或 synthetic queue 测试，论文中只能记为实现进展，不能写成已验证架构创新。
