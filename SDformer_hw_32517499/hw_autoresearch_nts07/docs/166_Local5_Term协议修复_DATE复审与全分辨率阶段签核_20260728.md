# Local5 Term 协议修复、DATE 复审与全分辨率阶段签核

日期：2026-07-28

## 0. 阶段结论

本阶段回答三个问题：

1. **Term 能否用于 Local5：能，而且已有可运行 RTL。**
   当前 Local5 使用逐 destination 的多重集 term：
   `{destination, lane, gate, multiplicity}`。Motion term 是
   `multiplicity=1` 的集合特例。
2. **TTB/STT 能否单列为架构贡献：当前不能。**
   现有实现仍主要是 bundle/descriptor。只有升级为真正控制 payload 驻留、
   exact skip、attention、term commit 和 retire 的可执行语义 tile，才能作为
   DATE 架构贡献候选。
3. **硬件应按 crop 还是全分辨率：最终论文必须按全分辨率。**
   RTL/DC 对象仍是可复用 tile engine，不是整帧空间展开；但整帧周期、SRAM、
   带宽、功耗和 FPS 必须按 DSEC `480x640` 累计。

本阶段还修复了一处真实的 P0 协议问题：

> Local5 multiset bridge 曾混淆 destination term 结束与 head/window 结束，
> EXPLODE 模式可能把一个 MFEP 输入项错误地跨成多个 DCTF term。

修复后完整 Local5 回归通过，但这只关闭了当前逐 destination term 的协议正确性，
**不代表 MPET、RES-Tile、共享 multiset DCTF 或 w15 PPA 已完成。**

---

## 1. 证据状态

| 项目 | 当前状态 | 证据等级 |
|---|---|---|
| Local5 逐 destination 多重集 term | 已实现 | `[RTL]` |
| multiplicity EXPLODE bridge | 已修复 term 边界 | `[RTL]` |
| bridge 反压稳定与零气泡切换 | 已验证 | `[RTL]` |
| MFEP 重复/越界方向原子拒绝 | 已验证 | `[RTL]` |
| Local5 score→gate→term→窗口顶层 | 已通过定向回归 | `[RTL]` |
| crop-w9/full-w9/full-w15 容量账本 | 已生成 | `[静态模型]` |
| Motion/Local5 统一 set/multiset schema | 已定义 | `[架构提案]` |
| MPET 跨 destination 聚合 | 未实现 | `[缺失]` |
| RES-Tile 可执行控制器 | 未实现 | `[缺失]` |
| 真正 multiplicity-aware DCTF executor | 未实现 | `[缺失]` |
| w15 真实 trace bit-exact | 未完成 | `[缺失]` |
| 同约束 DC/STA/SAIF | 未完成 | `[缺失]` |

---

## 2. 独立 DATE 预审

### 2.1 第一轮

独立 reviewer 子代理对 `docs/165`、Local5 bridge、MFEP 和分辨率账本进行检查，
给出：

| 维度 | 评分 |
|---|---:|
| DATE 当前准备度 | 2/5 |
| 当前新颖性 | 2/5 |
| 目标架构新颖性 | 3/5 |
| 证据完整度 | 2/5 |

主要意见：

1. Local5 term 方向成立，但 bridge 的 term 边界存在 P0 风险；
2. MPET 和 RES-Tile 仍是提案，不能写成已实现贡献；
3. 分辨率账本是容量模型，不是物理 SRAM/PPA；
4. 缺少 bridge→真实 DCTF fabric→executor 的端到端 scoreboard；
5. 缺少 w15 真实 trace、bit-exact 和目标工艺 PPA。

### 2.2 修复后复审

完成 P0 修复和定向验证后，独立 reviewer 给出：

| 维度 | 评分 |
|---|---:|
| DATE 当前准备度 | 3/5 |
| bridge P0 正确性 | 4/5 |
| 语义严谨性 | 4/5 |
| 证据边界 | 4/5 |
| 当前实现完整度 | 3/5 |
| 当前新颖性 | 2/5 |
| 提案新颖性 | 3/5 |

该评分说明：

- 协议可信度明显提升；
- 文档已经能区分当前 RTL 与提案；
- **架构新颖性没有因为修复协议而自动提升**；
- 想达到 DATE 可接收水平，仍要实现和量化 MPET、RES-Tile 或等价的系统级机制。

---

## 3. P0：Local5 term 边界问题

### 3.1 正确的三层边界

Local5 数据流至少有三种不同边界：

```text
beat_last:
  multiplicity EXPLODE 后，当前 DCTF command 是否为该 MFEP 输入项的最后一拍

term_last:
  当前 destination 的 MFEP term 流是否结束

head_last:
  当前 head/window 是否结束
```

对当前 bridge 而言，一个 MFEP 输入项就是一个 DCTF term：

```text
MFEP item = {dest, lane, gate, multiplicity}

SIDE_BAND:
  一个 item -> 一个 command -> 一个 DCTF term

EXPLODE:
  一个 item -> multiplicity 个 command beats
             -> 仍然只能是一个 DCTF term
```

因此：

```text
cmd_term_first = 当前 item 第一拍
cmd_term_last  = 当前 item 最后一拍
cmd_head_last  = 输入 head_last 且当前 item 最后一拍
```

输入侧 `term_last` 只表示该 destination 的最后一个 MFEP item，不能直接当作
每个 DCTF term 的结束信号；`term_head_last` 表示该 item 同时是 head/window
最后一项。

### 3.2 已完成的修复

涉及文件：

- `rtl_local5/local5_dctf_multiset_bridge.sv`
- `rtl_local5/local5_window_attention_top.sv`
- `rtl_local5/local5_linebuf_window_top.sv`

修复内容：

1. bridge 增加独立 `term_head_last` 输入；
2. `cmd_term_first/cmd_term_last` 按一个 MFEP item 的拍内边界生成；
3. `cmd_head_last` 只在最后 item 的最后 multiplicity beat 置位；
4. `issue_seq` 每个 MFEP item 加一；
5. `sequence` 每个实际输出 command 加一；
6. 拒绝非法组合 `term_head_last && !term_last`；
7. 顶层用真实 `sgt_term_last`/`sgt_tl` 连接 destination term 边界；
8. 顶层用 `destination-last && head-last` 生成 bridge 的 head/window 结束。

该修复保持当前 EXPLODE 数值语义，不宣称已经得到 MPET 的 command 压缩收益。

---

## 4. MFEP 输入协议硬化

涉及文件：

- `rtl_local5/local5_mfep_term_builder.sv`
- `tb_local5/tb_local5_mfep_protocol.sv`

原实现接收 `edge_dir`，但没有把方向作为协议唯一性条件。现在增加：

1. 当前 destination 内的方向 seen-mask；
2. 重复方向原子 abort；
3. 超出 `N_CAND` 的方向原子 abort；
4. 临时候选数组由固定 `[0:4]` 改为 `[0:N_CAND-1]`。

这些修改避免同一 self/N/S/E/W 候选被重复计数，保护 multiplicity 的精确语义。

---

## 5. 新增验证

### 5.1 Bridge 协议 TB

文件：

```text
tb_local5/tb_local5_dctf_multiset_bridge_protocol.sv
```

覆盖：

- multiplicity `3/1/2`；
- `cmd_term_first/cmd_term_last/cmd_head_last`；
- command sequence 和 issue sequence；
- 确定性反压；
- `valid && !ready` 时所有输出字段稳定；
- 两次 final beat 与下一 term 的零气泡切换。

结果：

```text
PASS tb_local5_dctf_multiset_bridge_protocol beats=6 zero_bubble=2
```

### 5.2 MFEP 协议 TB

文件：

```text
tb_local5/tb_local5_mfep_protocol.sv
```

覆盖：

- 重复方向；
- 越界方向；
- 原子 abort，不允许输出部分错误 term。

结果：

```text
PASS tb_local5_mfep_protocol
```

### 5.3 完整 Local5 回归

入口：

```text
sim_local5/run_local5_parity_checks.sh
```

最终日志：

```text
build_local5/parity/final_regression_term_tile_postreview_20260728.log
```

关键结果：

| 测试 | 结果 |
|---|---|
| score/Shiftmax 金向量 | 256 vectors PASS |
| row context | 96 edges / 24 rows PASS |
| TARE row context | 52 issues PASS |
| MFEP term builder | 44 terms PASS |
| MFEP sparse-last | PASS |
| MFEP protocol | PASS |
| multiset bridge protocol | 6 beats、2 次零气泡切换 PASS |
| score→gate→term | 460 commands / 8 vectors PASS |
| line buffer | PASS |
| window4 | 4 dest、219 commands、420 cycles PASS |
| zero-term window | 52 cycles PASS |
| window16 | 16 dest、883 commands、1662 cycles PASS |
| line-buffer window | 3 windows，均值 1118 cycles PASS |

最终：

```text
ALL LOCAL5 PARITY+WINDOW CHECKS PASSED
```

---

## 6. Term 的统一架构语义

### 6.1 当前共同式

Motion 与 Local5 都可以写成：

```text
Acc[d,o] += multiplicity * gate * W[lane,o]
```

其中：

```text
Motion: multiplicity = 1
Local5: multiplicity = 1..5
```

这给出统一 schema：

```text
Term = {
  group/head/input-channel/output-tile,
  gate,
  lane,
  multiplicity,
  segmented destination set
}
```

当前 Local5 RTL 只实现到“一个 destination 一个 multiset term”。目标 MPET 需要把
同一完整 key 的多个 destination 聚为一个 destination set：

```text
(gate, lane, multiplicity) -> destination bitmap/list
```

### 6.2 MPET 的价值判据

MPET 不能只靠命名成立，必须相对以下基线量化：

1. per-edge dense projection；
2. 当前 per-destination MFEP；
3. multiplicity EXPLODE DCTF；
4. 真正 multiplicity-aware DCTF。

至少报告：

- term/command 数；
- product 计算数；
- weight read 数；
- destination write 数；
- fabric stall；
- accumulator bank conflict；
- cycle、area、power、EDP；
- bit-exact mismatch。

若真实 Local5 trace 的 `(gate,lane,m)` 跨 destination 复用不足，MPET 不晋级为主贡献。

---

## 7. TTB/STT 如何升级

### 7.1 当前边界

当前 TTB/STT 的主要能力是：

- token/time bundle；
- phase sideband；
- 空 tile 或 K-zero 的统计/门控依据；
- line-buffer/window 的描述。

这些能力本身属于常规调度和 metadata，不足以单列 DATE 架构贡献。

### 7.2 目标 RES-Tile

TTB/STT 只有承担以下可执行职责后，才升级为：

> Resolution-Elastic Semantic Residency Tile，RES-Tile

```text
RES-Tile descriptor
  -> 决定 Q/K payload 是否读取与驻留
  -> 选择 Motion temporal-pair 或 Local5 3-row neighborhood
  -> 执行 exact empty/K-zero/TARE 门控
  -> 触发 score 与 normalization
  -> 原子提交 set/multiset term
  -> 等待 term retire 后释放 tile
```

它的价值要通过以下三组消融证明：

| 基线 | 对比目标 |
|---|---|
| descriptor-only | 是否只减少控制位，无实际数据搬运收益 |
| metadata-first gating | 是否减少 payload fetch |
| full RES-Tile | 是否进一步减少中间物化、仲裁和生命周期气泡 |

TTB/STT 和 term 的边界应保持清晰：

```text
RES-Tile = 输入驻留、数据获取、计算生命周期
Term     = 输出复用、投影发射、原子退休生命周期
```

它们由 exact tile-to-term transducer 连接，而不是同一个概念换两个名字。

---

## 8. Crop 与全分辨率口径

### 8.1 为什么最终必须使用全分辨率

光流部署的真实输入是 DSEC `480x640`。crop 会改变：

- window 数和边界比例；
- partial window/padding 比例；
- Local5 邻域边界无效边数；
- descriptor 数；
- SRAM 容量和端口压力；
- 片上/片外带宽；
- 全帧周期、功耗和 FPS。

因此最终主表必须使用全分辨率 trace 和整帧累计。

### 8.2 为什么 DC 不需要展开整帧

硬件采用 tile/time multiplex：

```text
一套或少量 tile engines
  x 每帧 tile/window 数
  x 12 encoder blocks
```

DC 综合的是代表性 tile engine、共享后端和必要控制/存储接口，而不是为每个
图像位置复制一套组合逻辑。分辨率主要影响：

- counter/ID 位宽；
- buffer depth；
- descriptor 数；
- 执行周期；
- SRAM/带宽；
- FPS 与能耗。

### 8.3 三个参数点

静态账本：

```text
results/resolution_tile_term_ledger_20260728/ledger.md
```

| 参数点 | 分辨率 | tokens/row | rows/frame | scheduled slots/frame | padding |
|---|---:|---:|---:|---:|---:|
| crop-w9 | 288x384 | 162 | 6,720 | 1,088,640 | 9.5238% |
| full-w9 | 480x640 | 162 | 19,980 | 3,236,760 | 15.4710% |
| full-w15 | 480x640 | 450 | 6,720 | 3,024,000 | 9.5238% |

解释：

- `full-w9` 保持 162-token 行核，但整帧 descriptor/row 数显著增加；
- `full-w15` 用更大窗口降低 row 数，但单行容量增至 450；
- `full-w15` 的 token/destination ID 至少需要 9 bit；
- 当前大量 `162` 和 8-bit 默认值说明 w15 不是简单改一个参数即可。

当前建议：

1. `w9-162` 保留为兼容和早期 RTL 回归点；
2. `w15-450` 作为容量设计点；
3. 最终算法主点等待 full-resolution 软件结果；
4. PPA 主表必须使用与最终算法一致的参数点；
5. 若最终软件仍选 w9，也必须按 full-w9 的整帧工作量报告。

账本中的 bit-packed 数字是逻辑下界，不能直接当 SRAM macro 面积或功耗：

| 项目 | w9 | w15 |
|---|---:|---:|
| token bitmap | 21 B | 57 B |
| 单个 Q/K tile/head | 648 B | 1800 B |
| Q7 score 物化 | 162 B | 450 B |
| Q1.7 gate | 183 B | 507 B |
| SCS histogram | 35 B | 40 B |
| Local5 三行 K/head | 216 B | 360 B |

---

## 9. 当前 DATE 贡献边界

现在可以安全写成：

1. 已实现 Local5 exact anchor/TARE/Shiftmax/MFEP 的叶到窗口 RTL；
2. 已实现并修复逐 destination multiset term 的协议链；
3. 已建立 Motion set 与 Local5 bounded-multiset 的统一数学接口；
4. 已建立 crop/full-w9/full-w15 的系统容量口径。

现在不能写成：

1. 已实现 MPET；
2. 已实现 RES-Tile；
3. Motion/Local5 已共享完整 DCTF；
4. w15 已 bit-exact；
5. 已完成 DC/STA/SAIF；
6. 当前硬件已经达到 DATE accept 水平。

推荐目标贡献结构仍是：

| 候选贡献 | 当前状态 |
|---|---|
| C1 Semantic-Anchor Exact Residual Execution | 叶/窗口 RTL 已有 |
| C2 Resolution-Elastic Semantic Tile Orchestration | 提案 |
| C3 Exact Set/Multiset Tile-to-Term Transduction | 提案 |
| C4 Term-Stationary Polymorphic Projection Fabric | 提案 |

C2-C4 在实现前应作为一条系统提案描述，不能拆成三条“已完成创新”。

---

## 10. 下一阶段硬门槛

按优先级：

1. **真实 trace term 复用审计**
   - 分别统计 Motion `(gate,lane)` 和 Local5 `(gate,lane,m)`；
   - 统计跨 destination 的 cardinality、run length、bitmap/list 密度；
   - 决定 MPET 是否值得实现。
2. **真正 multiplicity-aware backend**
   - payload 原生携带 `multiplicity`；
   - bank 内只计算一次 `m*gate*W`；
   - 不再用 EXPLODE 作为最终实现。
3. **bridge→fabric→executor scoreboard**
   - 随机反压；
   - term 原子性；
   - exactly-once；
   - 与 dense gated-K 整数金参考逐项等价。
4. **RES-Tile 最小执行切片**
   - payload fetch gating；
   - residency ownership；
   - Motion/Local5 模式；
   - term commit/retire。
5. **w15 容量修复**
   - 9-bit token/destination ID；
   - segmented destination set；
   - 450 深度 score/gate/buffer；
   - 真实 partial-window 语义。
6. **同约束物理评估**
   - w9 与 w15 分开；
   - Central/EXPLODE/MPET 同 SDC、同 SRAM 规则；
   - DC、STA、SAIF、面积、功耗、EDP；
   - full-resolution mean/p95/p99 和 FPS。

只有第 1 项证明复用充足，并且第 2-6 项至少关闭主要证据链，Term/RES-Tile
才能从“有潜力的架构抽象”晋级为 DATE 主贡献。

