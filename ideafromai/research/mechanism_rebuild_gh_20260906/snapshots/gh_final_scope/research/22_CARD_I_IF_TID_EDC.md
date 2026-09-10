# Card I — Interface Sketch (TID-DeblurLoop + EDC-ΔFuse)

**日期:** 2026-09-06（Asia/Shanghai）  
**作者:** 调研 ← iscas_ssh 对齐 Card I  
**底稿:** `21_NOVELTY_BOOST_TOP5_FORKA.md`  
**Tree（落地 RTL 时）:** `/workspace/sdformer_c1c2star_grokbot/` only · `// GROKBOT NEW FILE -- iscas_ssh`  
**Fork:** **A only**  
**前置:** CFP/SCI glue +（进行中）BL+NL 辅先验；本卡 = **下一对主刀**  
**候补第三:** ResHTR-Refiner（见文末 stub，不进本波默认 RTL）

| Order | Module | Role |
|---|---|---|
| **I1** | `c1s_tid_deblur_loop` | TID-DeblurLoop — corr-free 迭代去模糊环 → wake / exact |
| **I2** | `c1s_edc_delta_fuse` | EDC-ΔFuse — 多尺度 Δ-feat × 低分辨 corr 融合 → exact_boost |
| (I+1) | `c1s_reshtr_refiner` | ResHTR — 候补；TMA/BiSAT 种子稳后再开 |

**硬规则**
1. 只增强三刀脊柱；**不**替换 OP-STW / HBG / OGEC / PRRC / exact / CFP / SCI。  
2. **禁止** 4D corr volume 大 SRAM（TID 的卖点就是 corr-free）。  
3. **禁止** MX3P / 纯二值换岛 / CIM。  
4. N_TILE 默认 **8**（对齐 Card G/H）。  
5. OpenROAD / µ²/mW / AEE：证明只用 RTL 计数；无联合仿真不报 AEE。

---

## Wiring map（叠在 G/H + CFP/SCI glue 上）

```
evt bins + flow_seed(TMA/BiSAT) ──► c1s_tid_deblur_loop ──┬─ deblur_wake ──► wake_merge / OP-STW
                                                           ├─ exact_pref  ──► CFP/OGEC side
                                                           ├─ dflow/flow_out ──► SCI / ResHTR(opt)
                                                           └─ flow_hat_next ──► (loop)

feat_t0/tn + corr_lo(ECP) + match_ok ──► c1s_edc_delta_fuse ──┬─ exact_boost ──► AND/OR CFP.exact_req
                                                              ├─ detail_hit  ──► OGEC assist
                                                              └─ mot_fuse    ──► refine-before-exact

c1s_ogec × c1s_prrc × c1s_exact_capture   ← still the exact enqueue spine
c1s_cfp_confgate / c1s_sci_cleanexit      ← glue already (or landing); I1/I2 旁路增强
BL-Veto / NL-STMFA                        ← 辅先验（iscas_ssh 落地中）；非 Card I 主刀
```

**推荐组合（TB 写死）：**
- `exact_en_to_capture = ogec.exact_en & (cfp.exact_req | edc.exact_boost | {N{tid.exact_pref}})`  
  （具体 OR/AND 策略消融；默认：**CFP 主门，EDC/TID 为 boost**）  
- `wake_merged |= tid.deblur_wake`  
- `capture_en &= ~sci.exact_hold`

---

## I1 — `c1s_tid_deblur_loop` (TID-DeblurLoop)

**一句话:** 用先验 flow 对 event bins 做 motion-compensate / deblur → 残差 Δflow → warm-start；**无** 4D corr volume。  
**近亲:** IDNet **ICRA 2024** / arXiv:2211.13726（ID + TID）；对照 E-RAFT/TMA（corr-heavy）。  
**增量:** 相对 PredExit（二元停）/ SCI（warp scrub）多一条 **corr-free 去模糊控制环**；咬 K3 exact + K1 wake。

### Parameters
```systemverilog
parameter int N_TILE   = 8;
parameter int FLOW_W   = 8;
parameter int BIN_W    = 8;
parameter int N_BIN    = 4;
parameter int N_ITER   = 2;              // ID multi-iter; TID 常用 1 + 时间推进
parameter logic [FLOW_W-1:0] TH_DRES = 8'd6;
parameter bit MODE_TID = 1'b1;           // 1=TID online; 0=ID same-batch iters
parameter bit ABLATE_NO_DEBLUR = 1'b0;   // 透传 raw bins
```

### Ports
```systemverilog
module c1s_tid_deblur_loop #(/* params */) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic                         iter_fire_i,
  input  logic [BIN_W-1:0]             evt_bin        [N_TILE][N_BIN],
  input  logic [FLOW_W-1:0]            flow_seed      [N_TILE], // ← TMA/BiSAT/OP-STW
  input  logic [FLOW_W-1:0]            flow_hat_prev  [N_TILE], // TID 上一拍
  output logic [BIN_W-1:0]             evt_deblur     [N_TILE][N_BIN],
  output logic [FLOW_W-1:0]            dflow          [N_TILE],
  output logic [FLOW_W-1:0]            flow_out       [N_TILE], // seed + ΔF
  output logic [FLOW_W-1:0]            flow_hat_next  [N_TILE],
  output logic [N_TILE-1:0]            deblur_wake,
  output logic                         exact_pref,              // |Δ| 大 → 偏好 exact
  output logic                         loop_valid
);
```

### 接线
| From / To | Signal |
|---|---|
| ← `c2s_tma_agg.agg_flow` / `c2s_bisat_agg.fuse_flow` | `flow_seed` |
| ← 寄存器回环 `flow_hat_next` | `flow_hat_prev` |
| → `c1s_wake_merge` | `deblur_wake` OR 进 merged wake |
| → `c1s_cfp_confgate` / `c1s_ogec_gate` | `exact_pref` 侧信道 |
| → `c1s_prrc_ledger` / `c1s_exact_capture_wrap` | 大 Δ → boost/hold 策略（与 SCI 协调） |
| → `c1s_sci_cleanexit`（可选） | `flow_out` / deblur 质量 |
| **禁止** | 4D corr SRAM 宏 |

### Prove-by
| Counter | Meaning |
|---|---|
| `cnt_deblur` | deblur 触发次数 |
| `sum_abs_dflow` | 残差幅度 |
| `cnt_exact_pref` | exact_pref 断言 |
| `cnt_wake` | deblur_wake popcount 累计 |
| Ablation | NO_DEBLUR vs ID vs TID：`exact_hit` / `wake_pop`（固定 PRRC） |

### 勿 overclaim
- 非 invent IDNet；非 Jetson 8 ms 当 ASIC。  
- ID vs TID 必须消融；MVSEC 弱项是算法数据问题，勿藏。  
- 勿用「也支持 corr volume」把卖点冲掉。

---

## I2 — `c1s_edc_delta_fuse` (EDC-ΔFuse)

**一句话:** 多尺度时序 **特征差分图** 与 **低分辨 corr** 自适应融合，在 exact 前 refine，抬 ECP/OGEC 细节。  
**近亲:** EDCFlow arXiv:2506.03512 (2025)；对照 TMA/E-RAFT corr；与 TID（corr-free）正交可并存。  
**增量:** CFP 有 conf、OGEC 有 match，仍缺 **Δ-feat 多尺度路径**；比 PredExit/NL-STMFA 更贴 exact 前 refine。

### Parameters
```systemverilog
parameter int N_TILE   = 8;
parameter int FEAT_W   = 8;
parameter int CORR_W   = 16;
parameter int MOT_W    = 8;
parameter int N_SCALE  = 3;              // 近似 {1,2,5} 风格尺度
parameter logic [MOT_W-1:0] TH_DETAIL = 8'd32;
parameter bit ABLATE_CORR_ONLY = 1'b0;
parameter bit ABLATE_DIFF_ONLY = 1'b0;
```

### Ports
```systemverilog
module c1s_edc_delta_fuse #(/* params */) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [FEAT_W-1:0]            feat_t0   [N_TILE],
  input  logic [FEAT_W-1:0]            feat_tn   [N_TILE], // warped / 邻帧
  input  logic [CORR_W-1:0]            corr_lo   [N_TILE], // ← ECP / 低分辨 corr
  input  logic [N_TILE-1:0]            match_ok,           // ← OGEC pre
  output logic [MOT_W-1:0]             mot_diff  [N_TILE],
  output logic [MOT_W-1:0]             mot_fuse  [N_TILE],
  output logic [N_TILE-1:0]            detail_hit,
  output logic [N_TILE-1:0]            exact_boost,        // → 与 CFP.exact_req 组合
  output logic                         fuse_valid
);
```

### 接线
| From / To | Signal |
|---|---|
| ← `c1s_ecp_qkv_predictor` / corr bank | `corr_lo` |
| ← feat / MW warp 邻域 | `feat_t0`, `feat_tn` |
| ← `c1s_ogec_gate` 前 match | `match_ok` |
| → `c1s_cfp_confgate.exact_req` 组合 | `exact_boost` |
| → `c1s_ogec_gate` | `detail_hit` 辅助 match |
| → exact 路径 | refine-**before**-`c1s_exact_capture_wrap` |
| 与 TID | 正交：TID 管 bins/去模糊；EDC 管 Δ-feat×corr |

### Prove-by
| Counter | Meaning |
|---|---|
| `sum_mot_diff` / `sum_mot_fuse` | Δ vs 融合能量 |
| `cnt_detail_hit` | 边界/纹理 tile |
| `cnt_exact_boost` | boost 次数 |
| Ablation | CORR_ONLY / DIFF_ONLY / FULL：`capture_cnt`, `detail_hit` |

### 勿 overclaim
- 非替换 OGEC；非报 ASIC 上打败 EDCFlow AEE。  
- **禁止** 再引入高分辨全对 corr（EDC 本意是避开它）。  
- 与 TID 叙事写清：一条 corr-free 环，一条轻量 Δ×低分辨 corr fuse。

---

## Suggested file drop（G/H 风格）

| Module | RTL | TB |
|---|---|---|
| TID | `rtl_c1star/c1s_tid_deblur_loop.sv` | `tb_c1star/tb_c1s_tid_deblur_loop.sv` |
| EDC | `rtl_c1star/c1s_edc_delta_fuse.sv` | `tb_c1star/tb_c1s_edc_delta_fuse.sv` |

顶层：薄侧接线进 `c1s_front_pipe`（exact/wake 路径）；**不要**为 Card I 重写贡献列表。  
**实现序:** I1 TID → I2 EDC（共享 exact_hit / grant 计数 harness）→（可选）ResHTR。

---

## 候补 I+1 — `c1s_reshtr_refiner`（stub only）

**一句话:** TMA/BiSAT 线性种子 + HF 残差 refine（ResFlow arXiv:2412.09105）。  
**何时开:** TID+EDC 绿且种子稳定后；抬 K1 residual wake。  
**端口摘要:** `seed_flow`←TMA/BiSAT；`res_wake`→wake_merge；`prrc_fine_lvl`→PRRC。完整口见 `21_*` §1。  
**本波默认不实现。**

---

## Letter hygiene（Card I）

**可说:** digital TID deblur loop + EDC-style Δ-feat fuse enhancing OGEC×PRRC→exact（and residual wake），under Fork A.  
**不可说:** first event-OF HW；first corr-free OF；EDCFlow/IDNet 精度数字；CIM；MX3P；OpenROAD 当贡献；把 BL/NL/ResHTR 写成与 TID/EDC 同等主刀（除非另开卡）。

**与 BL+NL:** 辅先验继续落地 OK；信件主叙事仍三刀 + Card I 两增强器。

---

## Done means（草图卡）

1. 本文件：`ideafromai/research/22_CARD_I_IF_TID_EDC.md`  
2. iscas_ssh 可直接按 I1→I2 开 skeleton + iverilog TB  
3. 每模块 ≥3 定向 case + 上表计数可读  

