# G1 投影集成顶层 RTL 与 direct/NMF 整数等价

**日期**：2026-07-14  
**接手**：Codex session `019f365d-6ed1-76b2-a993-6b652298d9d8`（额度耗尽后由本 agent 续作）  
**范围**：把已有叶模块连成 G1 projection 集成顶层，并完成 direct dense 与 NMF 目录路径的整数等价  
**状态**：集成顶层定向仿真通过；fallback 扩展未做；真实 ordered workload 仍受 GPU 训练占用阻塞

---

## 1. 本轮目标与完成度

Codex 停在「叶模块齐、集成顶层未连」。本轮完成：

| 项 | 结果 |
|----|------|
| `hitflow_g1_projection_top` | 已实现 |
| NMF → product → multicast → accumulator | 已连线 |
| bias-commit final 输出 | 已串入控制器 |
| direct vs NMF 整数等价 TB | **3 组用例 PASS** |
| 叶模块回归（multicast/product） | PASS |
| 真实 H67 ordered-trace profile | **未跑**（A800 仍被训练占用，未抢占） |

---

## 2. 集成数据流

```text
group + tokens{gate_q1.7, K[LANES]}
        |
        v
hitflow_nmf_g1_builder
  directory terms {gate, lane, dest_bitmap}
  (fallback 本切片只标记 overflow，不展开)
        |
        v
hitflow_gate_product_engine
  weight_req/rsp  (TB/外部 ROM)
  product {bitmap, OUT_TILE×int17}
        |
        v
hitflow_segmented_multicast
  per-bank token updates
        |
        v
hitflow_banked_accumulator
  product RMW → bias-commit → final
```

控制器状态：

```text
IDLE → RUN(stream tokens + pipeline) → WAIT_DRAIN → BIAS(0..TOKENS-1) → FINISH → DONE
```

---

## 3. 修过的正确性/仿真问题

1. **multicast `product_ready`**：空闲时不应要求 `bitmap!=0`，否则 `pipe_idle` 死锁。  
2. **product `term_ready`**：IDLE 且未呈现 term 时应 ready；仅对非法 present term 拒绝。  
3. **multicast 组合环**：`update_token_ids` 与 `update_ready` 拆成两个 `always_comb`，切断  
   `token_ids → acc_ready → mcast_ready → token_ids` 环（Verilator UNOPTFLAT / iverilog 零延迟死转）。  
4. **NMF `group_done_ready`**：顶层在 NMF 处于 DONE 时持续 ready，否则 NMF 无法回 IDLE，下一窗口卡死。  
5. **TB 多窗口握手**：`group_done_ready` 必须保持到 `group_done_valid` 拉低，控制器才能离开 `ST_DONE`。

---

## 4. 验证

### 4.1 等价定义

**Direct（软件金参考，TB 内计算）**：

```text
acc[t][o] = bias[t][o] + Σ_{l:K[t,l]=1} gate[t] * W[l][o]
```

**RTL 路径**：NMF 按 `(gate,lane)` 合并目的 bitmap → 一次乘权重 → 多播累加 → 每 token 一次 bias-commit 输出。

### 4.2 TB 用例（`TOKENS=6,LANES=4,SLOTS=4,OUT_TILE=2,BANKS=2,SEGMENT=2`）

| 用例 | 内容 |
|------|------|
| CASE1 | 同 gate 多 token 合并；gate=0 / K-zero 过滤 |
| CASE2 | 多 lane、多共享 gate |
| CASE3 | gate=256 与 int8 边界权重；唯一 gate 数 ≤ SLOTS |

命令：

```bash
cd hw_autoresearch_nts07
iverilog -g2012 -s tb_hitflow_g1_projection_top -o /tmp/tb_g1.vvp \
  rtl_hitflow/hitflow_nmf_g1_builder.sv \
  rtl_hitflow/hitflow_gate_product_engine.sv \
  rtl_hitflow/hitflow_segmented_multicast.sv \
  rtl_hitflow/hitflow_banked_accumulator.sv \
  rtl_hitflow/hitflow_g1_projection_top.sv \
  tb_hitflow/tb_hitflow_g1_projection_top.sv
vvp /tmp/tb_g1.vvp
# 期望：PASS: G1 projection top direct/NMF integer equivalence
```

统一入口（已扩展）：

```bash
./sim_hitflow/run_projection_g1_checks.sh
```

### 4.3 证据等级

| 声明 | 等级 |
|------|------|
| 小参数 G1 集成路径与 direct 整数一致 | **[仿] directed** |
| 生产参数 162×32×SLOTS=4 端到端 | **未跑**（本 TB 为缩小参数） |
| overflow/fallback 无损展开 | **未实现**（overflow 仅 sticky；用例保证 ≤SLOTS 唯一 gate） |
| Yosys/DC 面积功耗 | **未作为本轮 signoff**（全参 NMF 综合仍重） |

---

## 5. 新增/修改文件

| 路径 | 说明 |
|------|------|
| `rtl_hitflow/hitflow_g1_projection_top.sv` | **新增** G1 集成顶层 |
| `tb_hitflow/tb_hitflow_g1_projection_top.sv` | **新增** 整数等价 TB |
| `rtl_hitflow/hitflow_segmented_multicast.sv` | ready 语义 + comb 拆分 |
| `rtl_hitflow/hitflow_gate_product_engine.sv` | term_ready 语义 |
| `rtl_hitflow/filelist.f` | 加入 top |
| `sim_hitflow/run_projection_g1_checks.sh` | 加入 top 仿真/lint/yosys |
| `docs/74_...` | 本文 |

---

## 6. 明确不做 / 未完成

1. **fallback 路径展开**（overflow 时逐 token 直投）。  
2. **多 output tile 循环**（当前 `term_output_tile=0`，`OUT_CH=OUT_TILE`）。  
3. **global_input_channel = head_id*32+lane**（叶模块已预留，集成层仍用 head 内 lane）。  
4. **真实 H67 ordered workload / TTB-v2 profile**（GPU 训练占用中，watcher 仍等待，**未杀训练**）。  
5. **DC / SAIF / Formality / SRAM macro**。  
6. 参数冻结 G2/G4、蝶形互连——仍等真实统计交叉点。

---

## 7. 下一步建议（不重训）

1. GPU 空闲后跑 **ordered-trace profile v2**（pair_empty / kzero / motion / w_set 联合分布）。  
2. 用 profile 决定是否实现 **fallback 展开** 与 **SLOTS/SEGMENT** 冻结。  
3. 放大 TB 到 `TOKENS=162,LANES=32` 做随机向量等价（可 Verilator）。  
4. 再谈 dual-path / PHEA / 全 encoder 端口模型。

---

## 8. 一句话

> G1 投影后端已从「四片叶 RTL」推进到「可仿真集成顶层」，并在缩小参数下证明 **NMF 目录多播与 direct 累加整数一致**；系统架构与 DATE 主表仍缺真实 workload 事务统计和 DC 口径。
