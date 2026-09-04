# GateStack PLAN/COMMIT 双 Tag 控制面与单 Context 执行闭环

## 1. 本轮回答的问题

上一阶段已经有 output-tile scheduler、三种 decoder 和共享 projection，但它们仍是分散部件。本轮解决三个系统问题：

1. 谁拥有一次 replay transaction，谁负责阻止部分启动？
2. 跨 output tile 时，持久 payload 身份和单次 execution 身份如何分离？
3. scheduler、slot/cache、decoder、projection 和末次释放能否形成真实端到端闭环？

结论是：上述三个问题在单 context 范围内已经形成 RTL 闭环并通过定向数值验证。

## 2. 架构层次

```text
group descriptor
       |
       v
+---------------------------+
| Output-Tile Scheduler     |
| tile_tag = group_tag + i  |
+-------------+-------------+
              | head issue
              v
+---------------------------+
| Replay Control Plane      |
| PLAN -> Atomic COMMIT     |
| -> Dual-Tag Lifecycle     |
+----+-----------+----------+
     |           |
     |           +---------------------------+
     | projection commit                     |
     v                                       v
+------------------+               +--------------------+
| Head-Slot SRAM   |               | Descriptor Cache   |
| payload words    |               | resident terms     |
+--------+---------+               +---------+----------+
         | slot words                        | descriptor stream
         v                                   v
+-------------------------------------------------------+
| Route-Locked Replay Word Router                       |
| resident / IPD32W / RAW41                             |
+-------------------------+-----------------------------+
                          v
+-------------------------------------------------------+
| Real Three-Decoder Shared Projection                  |
| decoder -> TDR -> multicast -> product -> accumulator |
| -> bias -> final                                      |
+-------------------------+-----------------------------+
                          |
                          v
               backend done guard
                          |
                          v
                dual-tag lifecycle
                          |
              final tile release slot/cache
                          |
                          v
                 scheduler head done
```

## 3. PLAN 与原子 COMMIT

### 3.1 PLAN 必须无副作用

`gatestack_replay_plan_builder` 只读取：

- head-slot 是否存在、payload tag、CSR/RAW mode、payload bits、word count；
- descriptor cache 是否命中、cache tag、term count。

它输出不可变计划：route、payload/execution tag、head/tile 元数据、是否拥有 cache、是否需要 slot replay、replay start word、resident term/event count。PLAN 阶段不允许启动 decoder、不允许启动 slot replay，也不允许占用 lifecycle。

### 3.2 COMMIT 必须原子

`gatestack_replay_atomic_commit` 只有在以下资源同周期 ready 时才产生 commit：

- projection head-start；
- lifecycle session slot；
- 若需要 payload words，则包括 head-slot replay 和 route lock。

因此不存在“decoder 已启动但 lifecycle 未记账”或“slot 已读但 projection 未接收”的半事务。

## 4. 为什么必须有双 Tag

跨 output tile 驻留时，一个 head 的压缩 payload 不变，但每个 tile 是独立执行：

```text
payload_tag   = 压缩 payload / cache 身份，跨 tile 不变
execution_tag = 当前 output tile 执行身份，每 tile 递增
```

具体规则：

- resident/IPD/RAW decoder 完成时检查 `payload_tag`；
- projection/backend 完成时检查 `execution_tag`；
- final output 和 tile done 携带 `execution_tag`；
- 只有 `last_output_tile=1` 时释放该 head 的 slot；
- 只有该 head 真正拥有 resident cache 时才同时释放 cache。

如果继续复用单 tag，第二个 tile 会把同一 payload 错判为旧 execution，或迫使每个 tile 重写 cache tag，破坏 residency 的意义。

## 5. 单 Context 执行顶层接口

新顶层为 `gatestack_single_context_execution_top`。

### 5.1 上游输入边界

| 接口 | 内容 | 当前所有者 |
|---|---|---|
| `group_*` | head 数、首 output tile、tile 数、group tag | encoder descriptor scheduler |
| `payload_commit_*` | CSR/RAW payload 写入 head-slot | attention/SCS 前端 |
| `descriptor_fill_*` | resident descriptor 预填 | descriptor build/fill 通路 |

### 5.2 参数/结果边界

| 接口 | 内容 |
|---|---|
| `weight_req/rsp_*` | 以 execution tag、input channel、output tile 索引权重 |
| `bias_req_*` | 每 token 的 bias/requant 输入 |
| `final_*` | 带 execution tag 的 token×OUT_TILE 最终结果 |
| `group_done_*` | group 原始 tag 和聚合 error |

### 5.3 Descriptor fill 的两类来源

`descriptor_fill_*` 仍保留为 group 启动前的上游预填接口。group 执行期间，IPD decoder 会在 header1 校验后发出 term count，并把后续 term descriptor 原子 fork 到 projection 与 residency cache；超出 cache 深度的 head 由 fill adapter 自动 bypass，不反压 projection。

## 6. 端到端验证

### 6.1 场景

```text
2 input heads × 2 output tiles

head0: resident cache hit, payload_tag=0x6800
head1: tile0 IPD miss，自动回填；tile1 resident hit

tile0 execution_tag=0x7800
tile1 execution_tag=0x7801
```

head0 的 gate/token 贡献和 head1 的 gate/token 贡献在同一 accumulator 中叠加，再加 bias。16 个最终 token 均逐值检查，而不是只检查 done 信号。

### 6.2 结果

| 指标 | 值 |
|---|---:|
| tile/head session | 2/4 |
| resident hit / IPD miss | 3/1 |
| slot replay/release | 4/2 |
| cache release | 2 |
| projection head/term | 4/4 |
| bias/final | 16/16 |
| mismatch/protocol error | 0/0 |
| 正常 group Icarus/Verilator | 123/124 cycles |
| 含 missing-slot abort 总周期 | 133/134 cycles |

RAW 在该 full-top TB 中未重复加入，但真实三 decoder 的 RAW 路径仍通过小尺度 79-cycle 和 T162 529-cycle 回归。

## 7. 存储与综合口径

默认参数结构检查保留 memory，不将 SRAM 展开为 flop：

| 项 | Yosys 结构值 |
|---|---:|
| memories | 12 |
| memory bits | 378208 bit |
| generic cells | 4007 |
| `$mul` | 43 |

其中 32 个 `$mul` 对应 `OUT_TILE=32` 的 projection product lanes；新增 11 个来自 slot/cache 参数化地址的“变量×常数”表达式。它们预期在技术综合中化简，但当前 Yosys 预技术 IR 仍保留这些节点，因此必须在 DC/tech-map 后检查，不能提前宣传成零乘法器地址生成。

不能把这些值写成 PPA。正式 DC 必须：

1. 将 head-slot、descriptor cache 和 accumulator memory 映射到目标 SRAM macro 或明确的 memory compiler 模型；
2. 提供目标工艺 `.db`、时钟、IO delay、uncertainty、max fanout 和 operating corner；
3. 对相同 trace 生成 SAIF，分开报告 logic、memory、clock 和 IO power；
4. 完成 RTL-to-netlist LEC。

## 8. 本轮代码审阅

### 8.1 已修复

| 问题 | 修复 |
|---|---|
| lifecycle 活跃时 builder 已 idle，可能接受第二请求覆盖完成元数据 | 增加单 outstanding admission gate，直到 completion handshake 才重新开放 |
| slot replay 与 route lock 未共同纳入原子 reserve | `slot_reserve_ready = slot_adapter_ready && router_ready` |
| head-slot 两个索引 function 和参数 reset loop 触发 Erie 3 MUST | 改为常数乘法索引；只复位 valid，对 invalid metadata 返回 0 |
| descriptor-cache 同类 Erie 3 MUST | 同样修复，并通过多 context 回归 |
| backend 只校验 tag，未校验 head index/last | 新增 backend done guard |

修复后 head-slot、descriptor-cache、新控制面、新叶模块和新顶层 Erie 均为 0 error/0 warning。

### 8.2 仍未关闭

| 优先级 | 问题 | 风险 |
|---|---|---|
| P1 | 无双 context 仲裁 | 当前不能证明并发 window 隔离 |
| P1 | 无 H67 trace 驱动的默认 full-top 长回归 | 小 TB 不能证明真实 workload 的长期活性 |
| P1 | 无 DC/SDC/SAIF/LEC | 不能给出可投稿的真实 PPA 和时序 |
| P2 | bias/requant 独立位宽尚未冻结 | 影响最终面积和数值接口 |

## 9. 下一阶段

下一阶段不应先扩多核，而应按以下顺序关闭可用性和 DC 边界：

1. 用 H67 ordered trace 生成默认 162×32 full-top transaction replay；
2. 冻结 SRAM macro contract、SDC 和 DC 文件清单；
3. 检查 11 个常数地址乘法节点的技术映射结果；
4. 再评估双 context 是否带来吞吐收益，避免过早增加仲裁复杂度。

## 10. 结果路径

- 控制面报告：`results/gatestack_control_plane_20260716/report.md`
- 单 context 执行报告：`results/gatestack_single_context_execution_20260716/report.md`
- 主回归入口：`sim_hitflow/run_gatestack_single_context_execution_checks.sh`
