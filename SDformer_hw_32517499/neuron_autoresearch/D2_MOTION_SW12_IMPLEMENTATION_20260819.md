# D2 跨窗语义（h89/motion_sw12_overlap，stride-12 重叠滑窗 + 滚动分母）实现说明

日期：2026-08-19。算法实现 agent 交付，对应
`CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md` 的 D2（J1-J6）与
`CLAUDE_ALGORITHM_CONTRACT_QUEUE_20260818.md` 的 Local5 线验收。
实现完全保持 Swin 分窗 (2,15,15) 与全部模型参数不动（checkpoint 与 Motion
ep35 锚点直接兼容），窗口划分在**算子内部**完成，纯算子消融。

红线遵守：未启动任何训练/评测/GPU；bsa_attention.py 为纯追加
（0 删除行，见 §6）；未改 H82/H86/H67/Local5/h87/h88 任何既有路径
（含 D1 的 h87、D3 的 h88，逐函数源码不变确认）。

---

## 1. 算子模式（纯追加）

新 mode：`motion_sw12_overlap` / 别名 `h89`（bsa_attention.py 追加区内）。

窗口划分决策：**不动 Swin 分窗，在算子内按 field 行块重建重叠窗**。Swin
行序 row = (b·n_pairs+wd)·n_sw+s，field f = (b·n_pairs+wd) 覆盖 dense 行
[f·n_sw, (f+1)·n_sw)，故 **n_fields = batch // n_sw**；场坐标 = tile 坐标
(15×15) 拼接，stride-12 链在场上滑窗（x 链 3 窗、尾窗 clamp 到 6 宽），
窗成员 = 场上 15×15 矩形的 dense (row, tok) 键集（含 t·225 偏移）。

滚动分母（J1 硬约束）：每窗分母 Z = Σ 2^s（s = 规范融合式分数，Q7 网格
[0,162]），16bit 块分解 int64 精确（c = s>>4 ∈ [0,10]，v = 1<<(s&15)，
每块和 ≤ 900·2^15 ≪ 2^63；重组 Σ_c z[c]<<(16·c) 恢复 Σ 2^s）。leave/enter
项由**场坐标几何条带**直接给出（enter = 新窗右/下 12 宽带、exit = 旧窗左/上
12 宽带，均按场边界 clamp，首窗 enter 恒 0、末窗 exit 恒 0）——不能用成员
掩码相减（相邻窗 900 个 gather 位置布局不同）。闭环
z_roll[w] = z_full[0] + Σ_{i≤w} enter[i] − Σ_{i<w} exit[i]，
**与 z_full 逐窗逐块 `torch.equal`（J1 单测硬约束）**。

门还原（J3）：gate_w 按窗 shiftmax（行和 ∈ (0.5,1]，2 幂舍入）→
g_final = scatter_add 到 dense 键 → gate = g_final/mult（mult = 重叠重数），
Σ_t g_final(t) == #windows 在精确有理数重算下恒等。

规范融合式分数（与 D1 同式）：s = min(RNE16(64·o_t + sz_t + 16·m̄), 162)，
m̄ = popcount(K_0 ⊕ K_1) 两切片共享，`binary_motion_xor_alpha != 0` 抛
RuntimeError（运动不双重计数）。

## 2. 新增函数（bsa_attention.py 追加区，7 个）

1. `_d2_overlap_chain(total, wsize, stride)`：一维重叠链 [(start, end)]，
   尾窗 clamp（(30, 15, 12) → [(0,15),(12,27),(24,30)]）。
2. `_d2_decompose_field_batch(batch_total, n_pairs, cfg)`：attention batch
   分解为 (B, n_pairs, n_sw)；候选 n_sw ∈ (1376, 352, 88, 24, 6, 2)
   （`_D1_SPATIAL_WINDOW_CANDIDATES`，w15 全分辨率族），首个整除解采用；
   `sw12_batch` 是偏好而非覆盖——评测 bs1 时 batch 变化自动回退。
3. `_d2_field_grid(n_sw, cfg)`：(1,2)/(2,3)/(4,6)/(8,11)/(16,22)/(32,43)
   tile 网格（`_D2_FIELD_GRID_BY_WINDOWS`）。
4. `_d2_overlap_window_plan(n_y, n_x, wsize, stride, device)`：重叠窗几何
   计划（ys/xs、n_ow、row_idx/tok_idx/valid [n_ow, 450]、mult [n_sw, 450]、
   entry_band/exit_band 条带掩码）。
5. `_d2_pow2_chunk(win_scores)`：16bit 块分解 (c, v)，int64 精确。
6. `_d2_exp_flow_ledger(...)`：J6 流量账（与 check_d2 逐式一致）。
7. `_d2_catalog_bands(plan, scores_field)`：J5 跨窗目录（相邻窗共享带身份码
   flat=(t·field_h+y)·field_w+x + 类码表）。
8. `_binary_motion_sw12_overlap_attention(q_orig, k_orig, cfg)`：主算子，
   返回 (attn, row_sum, gate, sw12_stats)。sw12_stats 含 scores / rolling_z /
   z_full / exp_ledger / catalog / window_plan / gate_final / gate_mult /
   batch_decomposition / window_counts。

## 3. 挂载账本（forward 验证用，均 .detach()）

`_h9_d2_scores / _h9_d2_rolling_z / _h9_d2_z_full / _h9_d2_exp_ledger /
_h9_d2_catalog / _h9_d2_window_plan / _h9_d2_gate_final / _h9_d2_gate_mult /
_h9_d2_batch_decomposition / _h9_d2_window_counts`。

## 4. 实现期发现

1. **双 gather 成员错配（真实算子 bug，单测抓到）**：`sc.gather(2, row).
   gather(3, tok)` 两级 gather 各自按自己的位置 k 取索引，等价于
   (row[tok[k]], tok[k])，与 (row[k], tok[k]) 成员错位（J1 逐位 FAIL 的
   根因）。修复：展平 (row·450+tok) 单次 gather（与门 scatter 的 win_key
   同键）。修复后 J1 逐位 `torch.equal`、J3 Fraction 精确守恒全过。
2. 测试侧修正：h67 参考分数测试在 float 张量上做位运算（`q & k0` 报
   RuntimeError，改 int64）；`preserve_mean` 默认 True 会整体放大 gate，
   mean·mult 恒等测试须在 preserve_mean=False 下进行；gate_final 幂和界
   用理论界（相对误差 < 0.5）而非经验 0.35。
3. 滚动链 enter/exit 不能用成员掩码相减：相邻窗 900 个 gather 位置布局
   不同（tile 边界分裂），必须用场坐标几何条带。

## 5. 测试（CPU，unittest，全绿 38 例）

- `tests/test_motion_sw12_overlap_scores.py`（24 例）：算子级 J1-J6 ——
  J1 滚动分母逐位（torch.equal + Python int 全量重算 + 16bit 块无溢出）、
  J2 共享带身份（几何 (1,2) 网格、mult 180/900 双覆盖、身份码 == 场
  flat 下标）、J3 门守恒（Fraction 精确 Σ==#windows、float 幂和界、
  mean·mult == g_final）、J4 类集下界、J5 目录贡献、J6 流量账
  （520/825/450/270/234000/222750/−4.8%/+58.7% 与 check_d2 逐式一致）、
  布局与 batch 分解、规范融合式分数逐位（含 m̄=0 与 check_d2 的
  h67_slot_score 一致）、config 解析、stride=15 退化解。
- `tests/test_motion_sw12_overlap_forward.py`（14 例）：注入式
  `_qk_shiftmax_gate_forward`（Identity 子模块 + 二值 x，窗口 (2,15,15)，
  B*=20 行 = 2×5×2）端到端 —— F1 形状、F2 mult 加权守恒（Σ g_final
  ∈ (0.5·n_ow, n_ow]）、F3 账本挂载 + **forward 级 J1 逐位复验**、F4
  校验（motion alpha RuntimeError / num_steps ValueError / stride15 退化
  mult 全 1）、F5 回归（h60/h87/h88/h82/compat 均运行）、F6 STE 梯度。
- 既有回归：D1（29 例）+ D3（28 例）全绿；check_d2 参考入口 ALL PASS。

## 6. 回归与 diff 纪律

diff vs 原始文件（/tmp/bsa_attention_pristine_20260819_d2.py）：
**0 删除行 / 633 追加行**。config 纯追加字段：`sw12_window_size`（默认 0 =
15）、`sw12_stride`（默认 0 = 12）、`sw12_num_steps`（默认 0）、`sw12_batch`
（默认 0 = 自动）、`sw12_window_grid`（默认 (0,0) = 按候选自动），含
config_from_dict 对应解析。h82/h86/h67/Local5/h87/h88 既有函数源码逐字节
不变（diff 无 `<` 行佐证）。

## 7. 配置（configs/generated/，manifest 已记录 sha256）

- `dsec_fullres_w15_H89_motion_sw12_overlap_ft5_short_20260819.yml`：short
  验证。n_epochs=5、bs 2、**seed 0**（红线）、force_save [4]；mode h89、
  sw12_window_size/stride/num_steps/batch = 15/12/10/2、
  binary_motion_xor_alpha 0.0；窗口 [2,15,15]；**lr 5e-5**（D1 教训：
  contract change 用保守学习率，不沿用模板 1e-4）；续训起点 = Motion ep35
  锚点（--prev_runid + --finetune 1）。
- `dsec_fullres_w15_H89_motion_sw12_overlap_ft40.yml`：fullres 模板。
  n_epochs=40、force_save [34,39]。
- `dsec_fullres_w15_H89_motion_sw12_overlap_manifest.json`：模板/锚点/两配置
  sha256。

对比口径（写入 NOTE 与 launcher 冻结合同）：**Motion 锚点 1.3297@ep35 基于
stride-15 稠密非重叠分窗，与 D2 重叠滑窗（窗口数 520→825）不同口径，AEE
数值不可直接比较**；合同验证以 h89 内部退化为准 —— stride=15（mult 全 1、
窗口数不增，= 稠密非重叠基线），pass 条件 = AEE(stride12) ≤ AEE(stride15)
·1.02；J1 逐位精确与 J3 门守恒为算子级硬约束，已在 CPU 单测逐位验证。

## 8. 启动步骤（GPU 空闲后，由队列协调者执行）

    python3 entrypoints/run_motion_sw12_overlap_short_20260819.py

launcher 纪律（仿 run_motion_t5_quotient_short_20260818.py）：fcntl 锁
/tmp/sdformer_h89_motion_sw12_overlap.lock + status.log + SHA 冻结合同
（算子 py / 配置 / 本说明 / 锚点 checkpoint）+ train.py subprocess。
前置检查：锚点 ep35 存在、配置存在、**GPU 空闲**（nvidia-smi memory.used
< 4096MiB 且无已知训练/评测进程；忙时记录并退出 0，不自动排队）。通过标准
（合同验证实验 1）：loss 曲线与 T=2 基线同量级（不塌、不减半退化），
step-1k 后单调下降；随后 fullres ft40 按 §7 对比口径验证
（AEE(stride12) ≤ AEE(stride15)·1.02）。
