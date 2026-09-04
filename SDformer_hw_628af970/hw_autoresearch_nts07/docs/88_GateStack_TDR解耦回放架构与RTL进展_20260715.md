# GateStack-TDR 解耦回放架构与 RTL 进展（2026-07-15）

## 1. 本轮结论

GateStack-IPD32W 主线新增一项系统微架构：

> **TDR：Term-Destination Decoupled Rendezvous，门码乘积与目标 token 解耦汇合。**

完整意图为：

```text
resident / sequential IPD32W / RAW41
  -> session-locked replay mux
  -> ordered term/event stream
  -> lossless term fork
       |- term -> decoupled gate*weight product engine
       `- event -> exact destination bitmap assembler
  -> tag + 13-bit issue-seq rendezvous
  -> 原 segmented multicast -> persistent accumulator
```

这解决了旧 G1 接口与新回放数据流之间的真实矛盾：旧 `hitflow_gate_product_engine` 必须在 term 握手时一次拿到完整 162-bit bitmap，因而无法让下一 term 的乘积生成与当前 term 的 token 解码/多播重叠。TDR 把两项工作拆开，并在进入原多播器前无损重组。

当前证据等级为 `[rtl叶级]`。TDR 叶模块全部可综合、仿真通过，但尚未形成单 head 三路径集成 top，不能把旧周期模型直接当作 RTL 实测加速比。

## 2. 为什么这是 workload 驱动的结构

H67 profile100 给出的约束为：

- token K-zero 约 88.7%；
- final-gate product work 减少 82.49%；
- IPD32W 比例 97.5015%；
- 每个 final-gate term 只需计算一次 `gate×weight tile`，随后传播到若干 token；
- descriptor residency Depth=80 在 CSR head 内命中 99.9826%。

因此瓶颈不是重复算同一个乘积，而是两种不等长工作流：

1. term 侧：权重请求和 `gate×weight`；
2. destination 侧：从 IPD token list 或 RAW record 恢复目标 token。

若二者串行，descriptor cache 节省的前端周期会被权重等待或 bitmap 建表吞掉。TDR 的目的就是让这两条链并行，并用严格顺序恢复原多播输入。

## 3. 本轮修正的存储接口

### 3.1 Head-slot 子区间回放

`gatestack_head_slot_sram_adapter` 新增：

```systemverilog
replay_start_word[WORD_INDEX_W-1:0]
```

语义为：

```text
0                                  -> RAW41 或顺序 IPD32W 全量回放
2 + ((term_count + 1) >> 1)        -> resident 命中的 token-only 回放
```

输出 `replay_word_index` 保持为子流内从 0 开始的相对序号，以兼容原顺序 decoder；物理 SRAM 地址使用 `slot_base + replay_start_word`。`start_word>=slot_words` 被拒绝并置 sticky protocol error。零 term head 不发起 token 回放。

### 3.2 不复制 event-total

Descriptor cache 继续保持 24-bit entry：

```text
{reserved2, destination_count8, lane5, gate9}
```

不额外保存 event-total。控制器从已校验的 slot metadata 精确推导：

```text
token_start_word = 2 + ((term_count + 1) >> 1)
event_total = payload_bits/8 - token_start_word*8
```

Resident joiner 再验证 `sum(destination_count)==event_total`。因此 doc87 的 Depth=80 存储模型没有因本轮接口增加而失效。

## 4. 新增 RTL

| 模块 | 职责 | 严格结果 | Yosys generic cell |
|---|---|---|---:|
| `gatestack_resident_replay_joiner` | 驻留 descriptor 与 token-only word 汇合；下一 term 预取 | PASS，Verilator 0 warning | 351 |
| `gatestack_raw_issue_adapter` | RAW direct event 转统一 term/event | PASS，Verilator 0 warning | 30 |
| `gatestack_replay_mux` | resident/IPD/RAW 按 head 锁存路径 | PASS，Verilator 0 warning | 99 |
| `gatestack_destination_bitmap_assembler` | 两项 term 缓冲、精确 token bitmap 重建 | PASS，Verilator 0 warning | 188 |
| `gatestack_decoupled_product_engine` | 无 bitmap 的 gate×weight/权重握手 | PASS，Verilator 0 warning | 79 |
| `gatestack_product_bitmap_join` | tag/issue-seq 双流汇合 | PASS，Verilator 0 warning | 38 |
| `gatestack_term_fork` | term 对 product/bitmap 两消费者无损分发 | PASS，Verilator 0 warning | 36 |

上述 cell 数只用于结构审计，不是工艺面积。Replay mux 的可变 packed part-select 已改成 elaboration-time unpack array，Yosys 中不再保留通用 `$mul`。Product engine 的 8 个 `$mul` 是默认 `OUT_TILE=8` 的真实 `gate×weight` 算术，不应和地址生成乘法混淆。

## 5. 顺序和数值合同

TDR 使用每 head 从 0 开始的 13-bit `issue_seq`：

- IPD/resident term 最多 128 项，13 bit 有余量；
- RAW41 最坏 `162×32=5184` 个 direct event，7 bit 不够；
- replay mux 以实际 term/event 握手生成统一序号，不依赖各 decoder 的局部 index；
- done 时必须满足 term 数和完成 event-term 数相等。

Bitmap assembler 检查：

- gate/lane/issue-seq 一致；
- first/last/head-last 相序一致；
- token 范围合法；
- valid 位数等于 event_count；
- 同 term token 无重复；
- 实际 destination 总数等于 descriptor 声明。

Product-bitmap join 只在 tag 与 issue-seq 同时一致时输出；不匹配项被丢弃并置 sticky protocol error，禁止静默错配。

RAW 和 IPD 的到达顺序可以不同，但最终 accumulator 必须使用足够宽的全精度累加，中间不饱和、不 requant，仅在所有 input head 和最后 output tile 完成后做一次 bias/requant。否则加法顺序可能破坏 bit-exact；该位宽合同仍待部署量化冻结。

## 6. 验证结果

本轮严格脚本覆盖：

- head-slot 全量/子区间/非法边界回放；
- resident descriptor 与 token 独立到达；
- term、event、word 三类独立反压；
- RAW 单 destination 统一适配；
- 三路径非选中 ready 恒为 0、route session 内稳定；
- bitmap 跨多 event beat 构建、term 预取和重复检查；
- product 权重请求/响应、正负边界乘积和输出反压；
- product 先到、bitmap 先到、错序丢弃；
- term fork 两消费者非同拍接受。

结果：

| 项 | 结果 |
|---|---:|
| 项目 Python unittest（`sdformerflow` 环境） | **77/77 PASS** |
| 本轮 9 个 RTL 严格脚本 | **全部 PASS** |
| Icarus 自检 | PASS |
| Verilator assertion | PASS |
| Verilator warning/error | **0** |
| Yosys `check` | PASS |
| Erie 独立 lint（7 个新模块） | **0 error，7 个 REC warning** |
| 本轮 GateStack 文件尾随空白扫描 | PASS |

仓库级 `git diff --check` 仍会报告训练侧既有文件 `neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md:9069` 的尾随空白；本轮未修改该处，也未把它计入 GateStack RTL 结果。

Erie 的 7 个 warning 均为循环边界或参数常量的“显式位宽建议”，不是 MUST error。已锁定的 `EVENT_WAYS=4`、`SOURCES=3` 循环改为字面常量后，新模块的 `FOR_CONST_BOUNDS` 错误已清零。修改过的旧 `gatestack_head_slot_sram_adapter` 仍有两个地址生成 `function` 和一个参数化 reset loop 被 Erie 的 Verilog-2001 生成规则判为 3 个 MUST error；Verilator/Yosys 对其均通过。该模块需在 replay-control 集成时无函数化，当前不得宣称 Erie 全包签核。

系统 Python 不含 torch，直接全发现会得到“74 项通过、1 个模块导入失败”；这不是算法测试失败。权威命令必须使用：

```bash
/opt/conda/envs/sdformerflow/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

## 7. 架构创新边界

可以作为 DATE 候选贡献组合表述：

1. **容量安全 IPD32W/RAW41 head stack**：压缩成功与 overflow 共用固定槽和后端，保持 exact。
2. **Profile-locked bounded descriptor residency**：用真实 head 分布锁定 Depth=80，命中只跳过 descriptor 前端，不删除计算。
3. **TDR 解耦汇合后端**：把 final-gate term product 与 token destination reconstruction 并行化，再按 tag/sequence 恢复原多播语义。
4. **表示异构、计算统一**：resident/IPD/RAW 只在回放表示不同，不复制 product/multicast/accumulator 核。

不能单独宣称“首次 product reuse”“首次 attention fusion”或“首次稀疏 bitmap”。Prosperity、FLAT、FuseMax 等已有相关一般机制。本文可辩护的差异是：H67 final-gate 语义、容量安全双格式、跨 output-tile descriptor residency 与 term-destination rendezvous 的联合架构及 bit-exact fallback。

## 8. 尚未完成的关键项

1. `gatestack_replay_control`：cache lookup、slot metadata、tag/mode/payload 联合校验和启动顺序。
2. 单 head 三路径 top：把 resident/IPD/RAW decoder、mux、fork、assembler、product、join、原 multicast/accumulator 接通。
3. 错误 drain：当前叶模块能检测错误，但完整 top 尚未证明错误流最终释放 cache/slot。
4. 完整 accumulator 位宽、bias、requant 部署合同。
5. 3/6/12/24 input head、全部 output tile 和双 context 生命周期。
6. ordered profile transcript、真实周期、VCD/SAIF 和目标库 DC/LEC。

## 9. 下一步及淘汰线

下一实现顺序：

1. 写 replay control，严格执行 `lookup -> meta -> slot begin -> tag/mode/payload check -> decoder start`。
2. 集成单 head、单 output tile 三路径 top，并与 Python IPD32W/RAW 金参考逐事件比较。
3. 将 TDR joined packet 接入原 segmented multicast，验证随机反压下 accumulator 100% bit-exact。
4. 用真实 profile head transcript 对比“旧串行 bitmap product”与 TDR 重叠周期。
5. 扩展多 head/tile 和双 context 后再生成 SAIF。

TDR 保留为论文贡献的最低门槛：

- 完整 ordered trace 相对公平串行后端周期改善至少 10%；
- GateStack 完整架构相对 direct 吞吐至少 1.20x；
- 同库同频、含 SRAM macro 的 EDP 改善至少 15%；
- 所有三路径 accumulator 输出 100% bit-exact。

当前判定：

> **TDR 已从概念推进到可综合叶模块和严格接口验证，但完整控制、数值 top 与 PPA 尚未闭合，仍不能直接交付可信 DC 主表。**
