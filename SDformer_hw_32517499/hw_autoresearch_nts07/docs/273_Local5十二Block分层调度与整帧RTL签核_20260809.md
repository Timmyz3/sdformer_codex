# Local5 十二 Block 分层调度与整帧 RTL 签核

日期：2026-08-09

## 1. 本轮结论

本轮完成 Local5 当前最高优先级的系统完整度缺口之一：把单个
`qfit_local5_projection_tile` 周围隐含的 stage/block/window/output-tile/input-head
循环显式化为一套可综合的 12-block 分层调度协议。

- `[rtl]` 一帧精确遍历 1320 个 `{stage, block, window}` 语义组；
- `[rtl]` 发出 6720 个输出 tile start request；
- `[rtl]` 发出 54000 次投影输入头 job/replay request；
- `[rtl]` 6720 个首访 decode/cache intent 与 6720 个末访 release intent 一一守恒；
- `[rtl]` Icarus 和 Verilator/SVA 在三组固定随机反压下账本一致；
- `[rtl]` 错误 completion tag 与重复 `start_frame` 被 fail-closed 捕获，错误后不再发出新作业；
- `[rtl]` Yosys 能读取并检查层次和 flatten 网表。

这不是 Local5 全 encoder 数值闭环。当前只证明控制顺序、作业身份和资源生命周期，
尚未证明 T450 token SRAM、权重 reload、跨输入头 Acc32 归约、bias/BN/residual 和
最终输出 bit-exact。

## 2. 为什么需要这一层

此前 Local5 已有以下单 tile 数据流：

```text
Q/K bit
  -> score + Shiftmax5
  -> relation transpose
  -> source-major term
  -> TCFM5 projection
  -> 单输入头/单输出 tile 的 Acc32
```

但它没有回答一帧如何覆盖 12 个 block，也没有回答同一组 relation/term 在不同输出
tile 间如何复用。若直接把 `6720` 当作“完整作业数”，会漏掉投影的输入头重放：
每个输出 tile 都必须遍历该 stage 的全部输入 head，因此实际投影回放总数是
`54000`。

本轮把三层工作量口径冻结为：

| 层次 | 公式 | 每帧次数 | 含义 |
|---|---|---:|---|
| 窗口语义组 | `sum(blocks_s * windows_s)` | 1320 | relation 的空间归属 |
| attention decode intent | `sum(groups_s * heads_s)` | 6720 | 每组各 input head 的首访意图 |
| 输出 tile start | `sum(groups_s * heads_s)` | 6720 | 每组按 head 数形成同数目的输出 tile 请求 |
| 投影输入头 job/replay request | `sum(groups_s * heads_s^2)` | 54000 | 每个输出 tile 请求遍历全部输入 head |

各 stage 展开如下：

| Stage | Block | Window/block | Head | 语义组 | 输出 tile | 输入头回放 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 440 | 3 | 880 | 2640 | 7920 |
| 1 | 2 | 120 | 6 | 240 | 1440 | 8640 |
| 2 | 6 | 30 | 12 | 180 | 2160 | 25920 |
| 3 | 2 | 10 | 24 | 20 | 480 | 11520 |
| 合计 | 12 | - | - | 1320 | 6720 | 54000 |

## 3. 调度数据流

```text
start_frame
    |
    v
12-block descriptor sequencer
  {stage, block, window, group_id}
    |
    v
group issue: tag = group_id * 32
    |
    v
shared output-tile scheduler
    | outer loop: output_tile = 0 .. heads-1
    |   |
    |   +-> tile_start(tag + output_tile)
    |   |
    |   +-> inner loop: input_head = 0 .. heads-1
    |         |
    |         +-> head_job
    |             decode_required = (output_tile == 0)
    |             cache_release   = (output_tile == heads-1)
    |             input_channel   = input_head * 32
    |         |
    |         +<- head_done(tag, input_head, error)
    |   |
    |   +<- tile_done(tag, error)
    |
    +<- group_done
    |
next window/block/stage or frame_done
```

输出 tile 在外层、输入 head 在内层，目的是让同一输出 tile 的 partial sum 保持驻留；
首输出 tile 负责建立 head-local relation/term cache，后续输出 tile 回放，末输出 tile
释放。当前 RTL 只发出这一生命周期合同，cache 本体和真实 weight context 尚未接入。

## 4. 双线复用边界

新顶层没有重新实现 output-tile/head 循环，而是复用 Motion 线已有的
`gatestack_output_tile_scheduler.sv`。Local5 只增加：

1. 12-block descriptor 几何；
2. group/window 递进；
3. Local5 的首访 decode 与末访 cache release intent 语义；
4. 整帧守恒计数与 fail-closed 检查。

这说明 Motion 与 Local5 可以共享“descriptor orchestration”控制骨架，但并不表示
两条线共享 attention 内核。Motion 仍是 H67 Motion-XOR/SCS/NMF 数据流，Local5
仍是 Shiftmax5/relation/TCFM5 数据流。

本轮调度器属于系统完整度基础设施，不单独包装成 DATE 创新。其作用是让后续
gate-stationary、relation memo、source-major term 等真正的数据流贡献能够在整帧
边界下被公平测量。

## 5. 验证结果

入口：

```bash
sim_qfit/run_qfit_local5_encoder_job_scheduler_checks.sh
```

生成报告：

```text
results/qfit_local5_encoder_job_scheduler_20260809/report.{md,json}
```

### 5.1 整帧随机反压

| 种子 | Icarus 周期 | Verilator/SVA 周期 | 结果 |
|---:|---:|---:|---|
| 1 | 309551 | 309551 | PASS |
| 44257 | 309666 | 309666 | PASS |
| 48879 | 310029 | 310029 | PASS |

每次均精确得到 `1320/6720/54000/6720/6720` 五项账本。周期来自 TB 的 1--4
周期作业服务延迟和随机 ready，不是 SRAM 周期、模型吞吐或 FPS。

### 5.2 SVA 与故障注入

SVA 覆盖：

- tile/head valid 在反压下保持身份与 payload；
- stage/block/window/output tile/input head 均在真实网络几何内；
- tag 的低 5 bit 与输出 tile 一致；
- decode intent 仅发生在首输出 tile；
- release intent 仅发生在末输出 tile；
- input channel base 等于 `input_head * 32`；
- frame_done 单周期；
- protocol_error 粘滞；
- 五项整帧计数不越界。

故障注入把首个 `head_done_tag` 翻转一位，并在另一轮运行中于 busy 状态重复发出
`start_frame`。Icarus 与 Verilator/SVA 均检测到 `protocol_error`；错误后的
tile/head 发射计数保持不变，且不产生 `frame_done`。

### 5.3 综合可读代理

| 检查 | 结果 |
|---|---:|
| Yosys hierarchy generic cell | 116 |
| Yosys flatten generic cell | 194 |
| `check -assert` | PASS |

`[rtl]` 这些数值只证明控制 RTL 可被开放综合工具读取。它们不是 DC 面积、STA
频率、SAIF 功耗或 ASIC PPA，不进入论文 PPA 主表。

## 6. 本轮负结果与修复

首版整帧 TB 在所有业务计数均正确时仍报失败。根因是 testbench 在 `frame_done`
NBA 更新后的同一个时钟沿读取软件计数，`done_pulses` 尚未由监视进程更新。这是
TB 调度竞争，不是 DUT 协议错误。修复为在下一上升沿后延迟一个 delta 再签核，
随后两个模拟器和三种子结果一致。

该问题也说明最终签核不能只看一个 `PASS` 文本；报告脚本会重新解析每个模拟器的
五项守恒计数，并要求同种子 Icarus/Verilator 完全一致。

## 7. DATE 证据边界

可以写：

> `[rtl]` Local5 的 12-block descriptor/control schedule 已在整帧几何下实现，
> 并在 54000 次 input-head job/replay request、三组随机反压、错误 tag 和重复
> start 注入中闭合。

不能写：

- “Local5 full encoder RTL 已闭环”；
- “每帧 309k cycle”是部署吞吐；
- “首访 decode 使 attention 计算减少到 1/heads”；
- “relation cache 已实现且无冲突”；
- “Yosys 194 cell 是 ASIC 面积优势”；
- “12-block 时间复用本身是 DATE 架构创新”。

## 8. 下一轮最高优先级

下一轮应在本调度器与现有单 tile 数值内核之间增加明确的 token/weight/result 服务
合同：

1. T450 Q/K 请求与返回携带 `{job_tag, plane, y, x}`，支持 1/2/4/随机有界 SRAM
   latency 与双向反压；
2. weight context 由 reset-only 改为显式 `load -> commit -> use -> release`，绑定
   `{stage, block, input_head, output_tile}`；
3. Acc32 结果通过 ready/valid 写出，所有结果接受且流水 drain 后才允许
   `head_done/tile_done`；
4. 在最终 checkpoint 前先用确定性非零 T450 oracle 闭合协议；checkpoint 到达后
   只替换 Q/K、mask、theta-folded weight 和 SHA；
5. Motion 同时维持现有回归，新机制只在 CPU profile/上界先过门后再写 RTL。

这一顺序先解决 Local5 “控制完整但数据服务仍悬空”的事实缺口，不扩张尚无真实
workload 支撑的低优先级机制。

## 9. 独立 DATE 评审与整改

### 9.1 首轮评审

独立只读审稿人给出 `3.0/5 Major Revision`：

| 维度 | 分数 |
|---|---:|
| 总推荐 | 3.0/5 |
| 新颖性 | 2.5/5 |
| 架构完整度 | 2.5/5 |
| 实现可信度 | 3.5/5 |
| 实验完整度 | 2.0/5 |

审稿人确认 `1320/6720/54000` 的静态拓扑算术正确，但指出：

1. 共享 output-tile scheduler 对错误 completion 只置错误位，仍可能继续发射，
   首版“fail-closed”声明不成立；
2. busy 状态重复 `start_frame` 的错误状态可能被同拍状态机赋值覆盖；
3. 6720 个 decode/cache 和 release 只是 intent，不是 cache 生命周期完成证据；
4. 54000 是 input-head job/replay request，不是 54000 次 SRAM、term、Acc32 已完成；
5. 下一轮必须连接带 tag 的 token/weight/result 数据服务，不应先扩新机制。

### 9.2 RTL 与证据整改

本轮针对前三项完成实质修复：

1. `gatestack_output_tile_scheduler` 在 head/tile completion tag、id 或 error
   不匹配时直接转入 `ST_GROUP_DONE(error)`，不再发射后续 head/tile；
2. Local5 frame scheduler 将 `start_conflict` 和下层 `scheduler_protocol_error`
   提升为高优先级错误分支，避免被正常 `case` 覆盖；重复 start 同步清空下层
   scheduler；
3. 新增 SVA：错误后不得出现 tile/head dispatch 或 frame_done，frame_done 必须
   对应 clean protocol；
4. 故障 TB 在错误后继续开放 ready 观察 12 周期，要求发射计数冻结；另新增
   busy double-start 场景；
5. 所有计数器和文档统一改称 `decode/release intent job`，54000 统一改称
   `input-head job/replay request`；
6. 共享 Motion 调度回归 `tiles=3/heads=9` 与四 stage sweep
   `groups=4/tiles=45/heads=765` 重新通过；
7. Local5 Icarus、Verilator/SVA 三种子整帧、两类故障注入、lint 和 Yosys
   全部重新通过。

第二项“数据服务空合同”没有通过措辞关闭，保留为下一轮唯一主任务。当前包仍不
能独立列为 DATE 创新，也不能把 intent/request 数量升级为真实执行量。

### 9.3 整改后关闭复审

同一独立审稿人重新只读检查 RTL、两套 SVA、故障 TB、最新日志和 `9/9` 源文件
SHA 后，给出：

- 包级 `Accept`，严格限定为“12-block 控制调度与错误隔离包”；
- completion mismatch 后停止发射：关闭；
- busy double-start 高优先级隔离：关闭；
- 错误后零新发射且无 `frame_done`：关闭；
- intent/request 语义边界：主体关闭；
- Local5 整线更新为 `3.1/5 Weak Reject / Major Revision`；
- 实现可信度由 `3.5` 升至 `3.8`，新颖性仍为 `2.5`，架构完整度仍为
  `2.5`，实验完整度仍为 `2.0`。

评分只因真实错误隔离和验证增强而上调，没有因文档数量、Yosys cell 或静态调度
循环加分。包级关闭不等于论文可接收，也不等于 Local5 数值整帧已完成。

关闭复审再次选择的唯一下一步是：让带 tag 的 Q/K token 返回、显式 weight
`load/commit/use/release`、Acc32 ready/valid 写回和流水 drain 共同决定
`head_done/tile_done`。在该闭环完成前，不扩新的 cache/scanner/scheduler 机制。
