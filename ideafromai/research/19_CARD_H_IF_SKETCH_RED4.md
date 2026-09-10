# Card H — Interface Sketch (4× RED knives)

**日期:** 2026-09-06（Asia/Shanghai）  
**作者:** 调研（Grok Bot）← iscas_ssh 派单  
**Tree（落地 RTL 时）:** `/workspace/sdformer_c1c2star_grokbot/` only · `// GROKBOT NEW FILE -- iscas_ssh`  
**Fork:** **A only**（OP-STW / HBG int8 *proposal* / OGEC×PRRC→exact_capture）  
**前置:** Card G（`c1s_tde3_prior` + `c2s_tma_agg`）绿后再开实现；本文件 = **端口级草图**，非已综合 RTL。  
**底稿:** `research/18_POST_TDE3_TMA_HARD_KNIVES.md`

**四把 RED（实现优先序）**

| Order | Module | Knife role |
|---|---|---|
| H1 | `c2s_bisat_agg` | BiSAT-Agg — 升级 TMA 双向时序织物 |
| H2 | `c1s_cfp_confgate` | CFP-ConfGate — conf → exact/PRRC 预算 |
| H3 | `c1s_sci_cleanexit` | SCI-CleanExit — scrub + exit → exact hold |
| H4 | `c2s_bui_guard_sdsa` | BUI-GuardSDSA — bit-guard → HBG gate |

**硬规则**
1. 只增强现有接线，**不**替换 OP-STW / HBG-RP / OGEC / PRRC / exact_capture。  
2. **禁止** MX3P / 纯二值 ATLIF 换岛；HBG payload 保持 int8 *proposal*。  
3. **禁止** CIM / 模拟硅声称；全部可综合数字。  
4. N_TILE 与 Card G 对齐默认 **8**（与 `c1s_tde3_prior` / `c2s_tma_agg` 一致）；顶层可参数化到 64。  
5. 证明只用 RTL 计数器 + 消融；**无** µ²/mW、**无** AEE（无联合仿真前）。

---

## Wiring map（现有模块端口名 = 真 RTL）

```
event / flow_hint ──► c1s_tde3_prior ──tde_wake──┐
flow_cur/prev+evt ──► c1s_op_stw_predictor ───────┼─► c1s_wake_merge ──wake_merged──► …
                      ▲                           │
         (H optional) │                           │
                      │                           │
feat slices ──► c2s_tma_agg ──agg_flow/early_exit/hyp_hint──► Motion-TTB / MFBD
                      ▲
                      │ fuse from H1 BiSAT
                      │
corr/match ──► c1s_ogec_gate ──exact_en/prop_en──┐
PRRC ledger ──allow_exact────────────────────────┼─► c1s_exact_capture_wrap
                      ▲                          │
                 H2 CFP conf                     │
                 H3 SCI exact_hold               │
                                                 │
ATLIF amp ──► c2s_hbg_rp_packetizer ──g/p/pe_clk_en──► ADP/SMAM
                      ▲
                 H4 BUI token_keep / payload_en
```

---

## H1 — `c2s_bisat_agg` (BiSAT-Agg)

**一句话机制:** 在 TMA 前向 split/lookup 上增加 **backward corr + SATMA 一致性融合**，输出 fused motion pack / occlusion-exit hint。  
**近亲:** BAT arXiv:2503.03256 (2025); TMA ICCV’23; VideoFlow ICCV’23（帧侧近亲）。  
**相对增量:** `c2s_tma_agg` 仅前向；本模块不删 TMA，作 **旁路升级**（`bwd_en_i=0` 时退化为 TMA-compatible）。

### Parameters
```systemverilog
parameter int N_TILE   = 8;
parameter int N_SLICE  = 4;      // match c2s_tma_agg
parameter int FEAT_W   = 4;
parameter int DIR_W    = 2;
parameter int HIT_W    = 4;
parameter logic [HIT_W-1:0] TH_FUSE = 4'd3;
parameter bit  ABLATE_FWD_ONLY = 1'b0;  // 1 = behave like TMA-only
```

### Ports
```systemverilog
module c2s_bisat_agg #(/* params */) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic                         bwd_en_i,              // 0 → fwd-only (TMA-compat)
  input  logic [$clog2(N_SLICE)-1:0]   slice_idx,
  input  logic [FEAT_W-1:0]            feat_fwd [N_TILE],
  input  logic [FEAT_W-1:0]            feat_bwd [N_TILE],     // ignore if !bwd_en_i
  input  logic [DIR_W-1:0]             dir_code [N_TILE],     // from TDE3 / TMA
  input  logic [FEAT_W-1:0]            tma_agg_flow [N_TILE], // seed from c2s_tma_agg
  output logic [FEAT_W-1:0]            fuse_flow [N_TILE],    // → Motion-TTB hyp / MFBD
  output logic [N_TILE-1:0]            fuse_hit,
  output logic                         occ_exit_hint,         // → OGEC / SCI
  output logic [1:0]                   hyp_hint_o,            // widen TMA hyp_hint
  output logic                         fuse_valid
);
```

### 接线
| From / To | Signal |
|---|---|
| ← `c2s_tma_agg.agg_flow` | `tma_agg_flow` |
| ← `c1s_tde3_prior.dir_code` 或 TMA `dir_code` | `dir_code` |
| → `c2s_motion_ttb_packer.hyp_id` / `c2s_mfbd.bundle_hyp` | `hyp_hint_o` 扩展 |
| → `c1s_ogec_gate` 侧信道（可选 OR 进 match） | `occ_exit_hint` |
| 旁路 | `bwd_en_i=0` 时 `fuse_flow≈tma_agg_flow` |

### Prove-by 计数（1 页）
| Counter | Meaning |
|---|---|
| `cnt_fwd_only` | bwd disabled cycles |
| `cnt_bwd_used` | bwd_en && fuse≠tma |
| `cnt_fuse_hit` | popcount(fuse_hit) sum |
| `cnt_occ_exit` | occ_exit_hint pulses |
| Ablation | FWD_ONLY vs FULL：`fuse_hit` / `hyp` 分布 |

### 勿 overclaim
- 非 “first temporal OF HW”；最多 **first digital BAT-style bi-agg under this spikeformer OF stack**。  
- BAT 为 GPU algo；本卡不报 DSEC AEE。  
- 剧烈抖动时 bwd 增益有限 — 消融里保留 FWD_ONLY。

---

## H2 — `c1s_cfp_confgate` (CFP-ConfGate)

**一句话机制:** 用 corr peak 生成 **tile 置信度**，驱动 `exact_req` / `prop_en` 与 **PRRC grant**（高 conf 早释配额，低 conf 加码 exact）。  
**近亲:** EMD-Flow / CFP, ICCV’23 DOI:10.1109/ICCV51070.2023.00873；GMFlow CVPR’22（遮挡传播近亲）。  
**相对增量:** OGEC 只有 match 二元；PRRC 有预算但无 conf 输入 — 本模块 **插在二者之间**。

### Parameters
```systemverilog
parameter int N_TILE   = 8;
parameter int PEAK_W   = 16;
parameter int CONF_W   = 8;
parameter int BUDGET_W = 8;
parameter logic [CONF_W-1:0] TH_HI = 8'd160;
parameter logic [CONF_W-1:0] TH_LO = 8'd64;
parameter bit ABLATE_ALWAYS_EXACT = 1'b0;
```

### Ports
```systemverilog
module c1s_cfp_confgate #(/* params */) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [PEAK_W-1:0]            corr_peak [N_TILE],
  input  logic [N_TILE-1:0]            match_ok,           // from upstream / OGEC pre
  input  logic [BUDGET_W-1:0]          prrc_budget_i,      // from c1s_prrc_ledger.budget slice
  output logic [CONF_W-1:0]            conf [N_TILE],
  output logic [N_TILE-1:0]            exact_req,          // → combine with OGEC exact_en
  output logic [N_TILE-1:0]            prop_pref,          // prefer propagate when conf high
  output logic [BUDGET_W-1:0]          prrc_grant,         // soft grant → spend policy
  output logic                         conf_valid
);
```

### 接线
| From / To | Signal |
|---|---|
| ← corr / ECP 峰 | `corr_peak` |
| ← `c1s_ogec_gate` 前 match 或并行 | `match_ok` |
| ← `c1s_prrc_ledger.budget` | `prrc_budget_i` |
| → 与 `ogec.exact_en` 组合：`exact_en_final = ogec.exact_en \| cfp.exact_req`（文档化 OR/AND 策略；默认 **AND 预算、OR 请求需消融**） | `exact_req` |
| → `c1s_exact_capture_wrap` 经 OGEC+PRRC | 间接 |
| → `c1s_prrc_ledger` 策略（可选 `spend_i` 加权） | `prrc_grant` |

**推荐默认组合（写进 TB）：**  
`allow_exact_final = prrc.allow_exact && (popcount(exact_req) <= prrc_grant)`  
`exact_en_to_capture = ogec.exact_en & exact_req`（低 conf 不进 exact）

### Prove-by 计数
| Counter | Meaning |
|---|---|
| `sum_conf` / `avg_conf` | quality meter |
| `cnt_exact_req` | tiles requesting exact |
| `cnt_prop_pref` | tiles preferring propagate |
| `cnt_grant_starve` | exact_req 但 grant=0 |
| Ablation | ALWAYS_EXACT vs CFP：`capture_cnt` vs `sum_conf` |

### 勿 overclaim
- Enhance OGEC，**不**替换；勿称 first occlusion OF。  
- conf 阈值需 ep34/算法侧标定后才能报 AEE。  
- 非 PredExit 整帧退出 — 是 **tile 级预算**。

---

## H3 — `c1s_sci_cleanexit` (SCI-CleanExit)

**一句话机制:** 每迭代用 warp 一致性 SCI 质量图做 **scrub（强制再更）/ exit_ok / exact_hold**，锐化 PredExit 并保护 exact 配额。  
**近亲:** SciFlow CVPRW EVW’24 arXiv:2404.08135；ERAFT ISCAS’25（PredExit 灵感，帧）。  
**相对增量:** PredExit = residual&lt;ε halt；本模块多 **scrub 通道** + 与 exact_capture 的 hold。

### Parameters
```systemverilog
parameter int N_TILE  = 8;
parameter int FEAT_W  = 8;
parameter int FLOW_W  = 8;
parameter int SCI_W   = 8;
parameter logic [SCI_W-1:0] TH_EXIT  = 8'd200;
parameter logic [SCI_W-1:0] TH_SCRUB = 8'd80;
parameter bit ABLATE_EXIT_ONLY = 1'b0;  // 1 = no scrub (PredExit-like)
```

### Ports
```systemverilog
module c1s_sci_cleanexit #(/* params */) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         it_valid_i,          // per refinement iter
  input  logic signed [FEAT_W-1:0]     f1 [N_TILE],
  input  logic signed [FEAT_W-1:0]     f2_warp [N_TILE],    // warp(F2,f_hat) 预计算或 stub
  input  logic signed [FLOW_W-1:0]     f_hat [N_TILE],
  input  logic                         bisat_occ_exit_i,    // optional from H1
  output logic [SCI_W-1:0]             g_sci [N_TILE],
  output logic [N_TILE-1:0]            scrub_en,
  output logic                         exit_ok,             // frame/tile-agg exit
  output logic                         exact_hold,          // 1 → freeze exact enqueue
  output logic                         sci_valid
);
```

### 接线
| From / To | Signal |
|---|---|
| ← 迭代前端 / MW / TMA fuse | `f1`, `f2_warp`, `f_hat` |
| ← `c2s_bisat_agg.occ_exit_hint`（可选） | `bisat_occ_exit_i` |
| → 迭代控制 FSM | `scrub_en`, `exit_ok` |
| → `c1s_exact_capture_wrap.capture_en` | `capture_en &= ~exact_hold` |
| → `c2s_hbg_rp_packetizer` 可选 | `pe_clk_en &= ~scrub_en_any`（质量差时停 MAC） |

### Prove-by 计数
| Counter | Meaning |
|---|---|
| `cnt_iter` | refinement iters |
| `cnt_scrub` | scrub_en pulses |
| `cnt_exit` | exit_ok asserts |
| `cnt_exact_hold` | hold cycles |
| `avg_g_sci` | quality |
| Ablation | EXIT_ONLY vs FULL scrub：`cnt_iter` / `capture_cnt` |

### 勿 overclaim
- SciFlow 为轻量 **帧** OF；claim = **SCI channel on event-spikeformer residual loop**。  
- RFL 仅训练 — letter 不卖。  
- 勿把 Snapdragon on-device 数字当本 ASIC。

---

## H4 — `c2s_bui_guard_sdsa` (BUI-GuardSDSA)

**一句话机制:** bit-serial score 不确定区间护栏 → **token_keep / payload_en**，无独立稀疏预测器；门控 HBG 的 gate 轨，payload 仅在幸存 token 上乘。  
**近亲:** PADE HPCA’26 / arXiv:2512.14322（BUI-GF）；SOFA MICRO’24；SpAtten HPCA’21。  
**相对增量:** EvQ-Win=几何邻居；STH/SP=mass/头分型；本模块管 **SDSA score 逐 bit 剪枝**，更贴 MAC 能耗。

### Parameters
```systemverilog
parameter int N_TOKEN = 8;       // align N_TILE or SDSA window
parameter int SCORE_W = 8;
parameter int BOUND_W = 8;
parameter logic [BOUND_W-1:0] TH_DROP = 8'd16;  // ub-lb width / margin
parameter bit ABLATE_KEEP_ALL = 1'b0;
```

### Ports
```systemverilog
module c2s_bui_guard_sdsa #(/* params */) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         bit_round_valid_i,
  input  logic [SCORE_W-1:0]           score_msb [N_TOKEN],  // partial bit-serial score
  input  logic [BOUND_W-1:0]           ub [N_TOKEN],
  input  logic [BOUND_W-1:0]           lb [N_TOKEN],
  input  logic                         hbg_g_i,              // optional AND with HBG.g
  output logic [N_TOKEN-1:0]           token_keep,
  output logic [N_TOKEN-1:0]           payload_en,           // ATLIF MAC enable
  output logic                         ooo_ready,            // issue when bounds decide
  output logic                         guard_valid
);
```

### 接线
| From / To | Signal |
|---|---|
| ← SDSA / SMAM 部分积（bit-round） | `score_msb`, `ub`, `lb` |
| ← `c2s_hbg_rp_packetizer.g` | `hbg_g_i` |
| → HBG 后级 / ADP-MAC enable | `payload_en`（**不**吸收 p 进纯二值） |
| → `c2s_smam_rp` / STH mask | `token_keep` |
| 语义 | `payload_en = token_keep & {N{hbg_g_i}}`（默认） |

### Prove-by 计数
| Counter | Meaning |
|---|---|
| `cnt_keep` / `cnt_drop` | sparsity |
| `cnt_payload_fire` | MAC enables |
| `cnt_ooo` | early decide cycles |
| Ablation | KEEP_ALL vs BUI：`payload_fire` ↓ 且功能计数不变（定向 TB） |

### 勿 overclaim
- PADE/SOFA 是 **LLM attn** HW；claim 限定 **BUI-guard on OF spikeformer SDSA**。  
- 勿称 first sparse-attn accelerator。  
- **禁止** 把 ATLIF payload 吸进纯 gate（Fork A 铁律）。

---

## Suggested file drop (when Card G green)

| Module | RTL path | TB path |
|---|---|---|
| BiSAT | `rtl_c2star/c2s_bisat_agg.sv` | `tb_c2star/tb_c2s_bisat_agg.sv` |
| CFP | `rtl_c1star/c1s_cfp_confgate.sv` | `tb_c1star/tb_c1s_cfp_confgate.sv` |
| SCI | `rtl_c1star/c1s_sci_cleanexit.sv` | `tb_c1star/tb_c1s_sci_cleanexit.sv` |
| BUI | `rtl_c2star/c2s_bui_guard_sdsa.sv` | `tb_c2star/tb_c2s_bui_guard_sdsa.sv` |

顶层：薄侧接线进 `c1s_front_pipe` / `c2s_back_pipe`；**不要**为 Card H 重写 18 模块动物园叙事。

**实现顺序建议:** H2 CFP（贴 exact 主刀）→ H3 SCI（控制环）→ H1 BiSAT（TMA 升级）→ H4 BUI（HBG 织物）。  
（若信件时空紧：H2+H3 优先于 H1/H4。）

---

## Letter hygiene（Card H）

**可说:** digital enhancers of wake / exact budget / temporal fabric / SDSA gate under Fork A.  
**不可说:** first event-OF HW；first sparse attn；CIM；MX3P；硅基 µ²/mW；无联合仿真的 AEE；18 模块 = 18 创新。

**EvQ-Win / PredExit-OF / BitHyp-Prior:** 降为消融基线，不进 Card H 实现队列。

---

## Done means（草图卡）

1. 本文件落在 `ideafromai/research/19_CARD_H_IF_SKETCH_RED4.md`  
2. iscas_ssh Card G 绿后，可直接按上表开 skeleton + iverilog TB  
3. 每模块 ≥3 定向 case + 上表计数可读  

