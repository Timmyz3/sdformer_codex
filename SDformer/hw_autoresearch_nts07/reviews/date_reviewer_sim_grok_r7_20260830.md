# DATE 模拟评审 r7（独立评审人 grok）

日期：2026-08-30 22:40 CST。接替 grok r6（2026-08-29 22:00，总分 3.6 / Weak Accept）。
评审对象：Motion / H67 ep35，Codex 约 M1008–M1278。
docs/359 SHA `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
不注入 Codex。不把 live decoder 73–74/120 写成完成。

现场：decoder `run_m1111dr2` PID 4122290 已跑约 18 h、约 74/120；无活 `dc_shell`/`simv`；无 `.tex`；git 仍 `c1531749`（8/25）。远端 ep34 训练中，最终 checkpoint **未选**。

---

## 0. 相对 r6：进度很大，可引用物理点几乎没动

r6 已经把 C1 3 ns setup、C2 等带宽 DC、C3 Fixed-T10 DC 记进 Table B。此后 24 h 编号冲到 M1278，但 **没有新的可进主表的 Synopsys 数字**。

| 线 | r6 已准入 | 本评新增 | 论文能否用 |
|---|---|---|---|
| C1 周期 | 1.746753× CPU same-ledger | 生产坐标更新为 **1.7591725402×**（10 sample / 812,160 task：434,242,823 vs 763,908,050） | 仍是 raw-CPU component opportunity |
| C1 面积 | 147,246 µm²，setup +0.0018 ns，九宏 | 无新 DC | 同 r6 |
| C1 权重服务 | 无 | M1199 II=2 makespan **1.734×** vs bit/zero（434.15M vs 752.97M） | **只是 weight-service schedule**，不是 RTL/全 C1 |
| C1 wrapper VCS | 核心 UNIT_DELAY PASS | R7–R12 连续隔离；R12 是 child-force seam 不是 RTL 多读 psum；R13 checker 未授权 VCS | **协议 wrapper 仍未闭合** |
| C2 面积/周期 | 1.017× / 4.541× 吞吐/mm² / −77.6% | 无新 DC | 同 r6 |
| C2 能量 | SAIF 源包 GO | 生产 mapped-gate **零 SAIF**；M1155/M1274 **STOP** 旧网表扩观察 | 功耗仍 false |
| C1 能量 | parent 2.04 vs 3.30 mJ，−38.2% | M1275：不得与 1.759× 融合成能效比 | 两行必须分列 |
| Decoder | D2/D3 10K 前缀 | 120-call 生产重放 **未完成**（74/120） | 不得当 Table-A |
| Checkpoint | ep35 冻结 | ep29/30/32 存在，ep34 进行中；`checkpoint_selected_now=false` | 硬件数字不得转移到未选 ckpt |
| Table-A | 0 行；component annex 有 C1/C2/C3 | 仍 0 行 | 系统 headline 不存在 |
| 正文 | 无 | 无 | Presentation 继续封顶 |

一句话：昨天已经够写 **Weak Accept 组件论文**；今天把验证和训练往系统门推，但 **系统门一扇都没关上**。把 24 h 合同增量当成硬件进度，会高估 Validation。

---

## 1. 审稿人现在会看到什么

### C1

可写：240 KiB 约束下 exact product-capture 的 CPU 1.759×；28 nm 九 1RW 宏 island 147.2 kµm²、3 ns setup MET；parent 组件能量 −38.2% vs all-write；核心 UNIT_DELAY 功能 VCS。

攻击：

1. 1.759× 仍非 RTL。II=2 权重 1.734× 是另一条服务轴，禁止与 1.759× 相乘或替换。
2. 147k 仍未覆盖 214,912 B 全存储。吞吐/mm² 拼表仍是拒稿级。
3. hold −0.09 ns / 9992 paths。
4. 今天整天的 R12/R13 证明 **集成协议 harness 还没绿**。审稿人若要 wrapper 级 VCS，你们拿不出 PASS，只能拿出「TB seam 误杀」的取证。这不是 RTL 错，但是 Implementation 扣分。

### C2

4.54× 吞吐/mm² 仍是最干净物理 headline。能量冲刺失败：license、UCLI、X 传播、watchdog，最后正确停在旧网表。**不要把 tiny 24 ns SAIF 写成生产功耗。**

### C3

无变化。完整性 island。

### 系统 / decoder / ckpt

decoder 74/120 健康，commit 禁止重试。完成后也只是 ep35 decoder-only diagnostic，`final_checkpoint_rebind_required=true`。最终 ckpt 未选。Table-A=0。这三条把 Strong Accept 锁死。

---

## 2. 打分

加权同 r3–r6：N/S/V 各 0.25，Fit 0.15，Manuscript 0.10。

| 维度 | r5 | r6 | Codex M1266 | **r7** | 说明 |
|---|---:|---:|---:|---:|---|
| Novelty | 3.4 | 3.5 | 3.5 | **3.5** | 无新对象差 |
| Soundness | 4.2 | 4.2 | 4.4 | **4.3** | R12 取证、不重跑、checker 止损；4.4 留给系统行也 fail-closed 的时候 |
| Validation | 3.3 | 3.6 | 2.8（他们把 Evaluation 单列） | **3.5** | 物理点与 r6 相同；wrapper VCS 与 PTPX 未转化；decoder 未完 |
| Fit | 3.8 | 3.8 | — | **3.8** | |
| Manuscript | 1.5 | 1.5 | 2.5（偏高） | **1.5** | 距 abstract 约 14 天，仍零页 |
| **加权** | 3.4 | 3.6 | ~3.35 | **3.5** | `0.25*(3.5+4.3+3.5)+0.15*3.8+0.10*1.5 = 3.545` → **3.5** |

相对 r6 的 3.6 **略降 0.1**：不是证据变假，是日历走了一天、可引用主表没长、Presentation 相对截止日期更差。Codex 自审 3.35 把 Evaluation 压到 2.8 合理（系统行），但把 Presentation 写成 2.5 不成立——没有六页就是 1.5。

录用：

| 形态 | 预期 |
|---|---|
| 无纸 | 无法审 |
| 现 Table B + 诚实六页（1.759× 标 model，C2 4.54× 吞吐/mm²，C1 147k setup） | **Weak Accept 45–55%** |
| decoder 120 完成且能标 [model] 补全网 ≥1.10× + 最终 ckpt 绑定 | Weak Accept 偏 Accept |
| 再加 PTPX 能量 + C1 RTL 周期桥 ≥1.50× | Accept 竞争 |
| 摘要 4.76× / 1.759× 当硅上 / tiny SAIF 当功耗 | Weak Reject |
| Strong Accept | **false** |

Reviewer-2 短评：

> Component evidence is unchanged in substance from yesterday: a 3 ns 1RW capture island and an honest equal-bandwidth area result. A day of protocol-VCS and power-tooling work produced sealed failures, not numbers. Until a decoder-complete, checkpoint-bound system row exists, this is a weak-accept component paper, not an optical-flow accelerator.

---

## 3. 口径红线（r7 增量）

r6 红线全部保留，另加：

- 禁止 1.759× × 1.734×；
- 禁止 M1045 2 kB / 24 ns SAIF 进功耗表；
- 禁止 74/120 或部分 JSONL 当 decoder-complete；
- 禁止把 R12 隔离包写成 wrapper PASS 或 RTL 功能失败；
- 禁止 ep34/未选 ckpt 继承 ep35 周期。

---

## 4. 真正还差的（按截止日期，不是按 M 号）

Abstract ~2026-09-13。从今天起：

1. **写六页。** 3.5 分硬件已经定型成组件论文。再拖 Presentation 会单独把 Weak Accept 打成 desk-ish reject。
2. **让 decoder 120 自然结束**，只收 diagnostic annex，不要重跑、不要并行第二条 18 h。
3. **等 ep34 + 四份 valid825 一次选择**；选出前不要重绑硬件数字。
4. **停止 C1 checker / R14 VCS 扩写。** 核心 VCS + 3 ns DC 已够 Table B。wrapper 限制用一句话写进 Implementation。
5. C2 PTPX 只有在 **新网表 + 新 one-shot** 且不挡写作时才做；旧链已 STOP。

Stop list：新 matcher、TDA RTL、用 147k÷1.759 报吞吐/mm²、把训练 loss 写成硬件 identity。
