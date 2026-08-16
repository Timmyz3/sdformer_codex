# C1 双 Workspace 共享 Serializer 架构与 RTL 实测

## 1. 架构结论

C1 已从调度模型升级为完整 RTL。它复制两个 canonical workspace，每个 workspace 各带一个小型组合格式策略，但只保留一套三格式 Serializer、atomic slot commit 和 slot SRAM 写口。输入 head 按到达顺序分配到空闲 workspace；每个 workspace 获得单调递增 sequence；共享后端只能选择 `sequence == next_emit_sequence` 的 metadata-ready workspace。

```text
final-gate/K stream
       |
       v
capture allocator + capture lock
       |                         sequence ROB(2 entries)
       +--> Workspace 0 -------> oldest-ready selector --+
       |                                                |
       +--> Workspace 1 -------> oldest-ready selector --+
                                                        v
                                      local combinational policy
                                                        v
                                       shared tri-format Serializer
                                                        v
                                           shared atomic typed slot
```

这不是双核计算复制。C1 的架构点是双 entry capture/service decoupling 与严格有序共享后端：下一 head 的 162-token 捕获和 canonical directory 构建可以与前一 head 的 descriptor/destination/RAW 序列化、commit 重叠，但后到 head 不能越序占用 Serializer。

## 2. 控制合同

### 2.1 Capture 分配

- `head_begin` 只在没有正在捕获的 head 且至少一个 workspace 空闲时 ready；
- 同一时刻只有一个 workspace 接收 `head_begin`；
- 从 `head_begin_fire` 到最后一个 `token_last_fire`，输入 token 固定路由到同一 owner；
- 两个 workspace 都被占用时，上游 head 在 `head_begin` 边界阻塞，不允许部分接收。

### 2.2 顺序发射

- 每次 head 分配时写入 `workspace_sequence`；
- `next_capture_sequence - next_emit_sequence <= 2`；
- 只有 sequence 等于 `next_emit_sequence` 的 workspace 可拉高 metadata ready；
- 正常 head 必须同时完成 workspace emit 和 Builder commit 才产生顶层 done；
- done 握手后 `next_emit_sequence` 加一，workspace 才回到可分配状态。

### 2.3 共享资源

C1 只实例化一套：

- `gatestack_typed_builder_commit_top`；
- `gatestack_typed_payload_serializer`；
- `gatestack_head_slot_sram_adapter`；
- slot commit/replay/release 控制。

两个 workspace 各自保留 RAW scratch、class directory、destination bitmap、局部 FSM 和小型组合格式策略。论文中不能把 C1 描述成双 Serializer，也不能把两个 workspace 及两个格式策略的面积遗漏。

## 3. 全 45-Head RTL 实测

验证范围为 `sample0/B0/window0` 四 stage 全部 45 个真实 head。每个 stage 内连续捕获并并发消费 done；stage 末逐 word replay 并 release，和此前 C1 模型的“stage 边界清空”口径一致。

| Stage | Head | C0 RTL | C1 模型 | C1 RTL | RTL-模型 | RTL 加速 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 1013 | 694 | 696 | +2 | 1.455x |
| 1 | 6 | 1032 | 987 | 994 | +7 | 1.038x |
| 2 | 12 | 3135 | 2152 | 2163 | +11 | 1.449x |
| 3 | 24 | 8898 | 6159 | 6182 | +23 | 1.439x |
| 合计 | 45 | 14078 | 9992 | 10035 | +43 | 1.403x |

RTL 相对模型只多 43 拍，误差 0.43%。因此 C1 收益已经从 `[模型]` 晋级为 `[rtl]`：总周期减少 28.72%。

数据完整性结果：

- 45 个 head；
- 861 个 slot word 逐 word 零失配；
- 762 个 term；
- 3226 个逻辑 destination；
- 2728 个 segmented-list/BPB work item；
- 0 aborted head、0 protocol error、0 顺序等待异常。

活动计数：capture/service 重叠 2356 拍；两个 workspace 均占用造成 capture 阻塞 2523 拍；workspace 输出背压 354 拍。

## 4. 面积与存储代价

| 开放综合结构代理 | C0 | C1 | 变化 |
|---|---:|---:|---:|
| generic cells | 3181 | 5576 | +75.29% |
| `$mem_v2` | 13 | 19 | +6 |
| `$mux` | 2097 | 3911 | +1814 |

C1 用 75% 左右的开放结构 cell 增量换取 40% 左右吞吐提升，尚不能证明 EDP 优势。这个数字包含未映射 SRAM 的大 mux 结构，不能替代 DC 面积。最终论文必须至少给出：

1. RAW scratch、bitmap directory、payload slot 分别映射 SRAM 宏后的面积；
2. C0/C1 同频率 SDC 下的 WNS/TNS；
3. 相同 45-head trace 的 SAIF 动态功耗和总能量；
4. C1 的 energy/head、area-normalized throughput 和 EDP；
5. 若 C1 EDP 不占优，则把 C1 定位为 throughput mode，C0 定位为 area mode。

## 5. 验证状态

已完成：

- 真实双 head IPD+FADC overlap，逐 word replay，Icarus PASS；
- 四 stage 45-head stage-bounded overlap，逐 word replay，Icarus PASS；
- 缩参双 head 控制路径，Icarus PASS；
- 全规模 C1 与所有绑定 SVA 的 Verilator lint/elaboration，0 warning；
- C1 顶层 Erie lint，0 error、0 warning；
- Yosys `check`，0 problem；
- workspace、Serializer、slot 叶模块已有独立 Verilator 动态 SVA 回归。

未完成且必须诚实披露：

- 全规模 C1 Verilator 动态执行因生成模型性能超过当前超时门限，没有动态 SVA PASS；
- 尚无随机 backpressure 长回归和覆盖率数据库；
- 尚无 DC/STA/SAIF、SRAM macro、mapped LEC；
- 尚无更大真实 trace 和 valid825 全部署等价。

## 6. DATE 贡献位置

C1 单独只是 ping-pong/double buffering，不能宣称新颖。它与本工作特有的 canonical gate/lane bitmap、三格式 residency、BPB 和 atomic typed slot 联合后，才形成可辩护贡献：

> 提出一种 sequence-ordered dual-workspace dataflow，在不复制 Serializer 的条件下，将 all-binary 事件 head 的捕获/分类与表示自适应序列化重叠，并保持 IPD/FADC/RAW 三种物理格式逐 word exact。

主贡献仍应组合表达为：

1. workload-derived canonical gate-stack 与 tri-format typed residency；
2. segmented exact walker + bitmap-preserving bypass，避免 scan 和 expand-then-rebuild；
3. sequence-ordered dual-workspace/shared-backend pipeline，隐藏 capture/analyze 延迟；
4. H67 all-binary 软件语义、真实 trace、RTL 和最终 PPA 的端到端协同证据。

## 7. 下一步门槛

下一步不是继续增加格式，而是降低 C1 的存储/互连代价并补目标库证据：

- 把 `class × lane × token` bitmap 从逻辑二维数组改成显式 class/segment bank，约束读写端口；
- 分离 RAW scratch 与 compressed directory 生命周期，评估是否可以共享或早释放；
- 生成 C0/C1 同约束 DC handoff，并接入真实 SRAM macro；
- 用 SAIF 回放同一 45-head trace；
- 扩充多 window trace，检查 1.403x 是否稳定；
- 再执行一次独立 DATE 审稿，届时只有真实 PPA/energy 和多 trace 仍缺时，才接近 Weak Accept。
