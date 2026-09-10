# R1+R2 全做合并计划（Codex 加速版）

**作者:** Grok Bot (`iscas_ssh`) — NEW FILE  
**日期:** 2026-09-05  
**用户决定:** R1 与 R2 **都做**（C1* 满配 + C2* 满配）；硬件实现预期用 **Codex** 提速。  
**纪律:** 不覆盖旧 RTL；新目录 `*_grokbot_*`；改代码前再向用户确认。

---

## 1. 范围冻结（全做 = 这些模块）

### C1* 满配
1. OP-STW predictor + wake dispatcher  
2. PRRC pyramid residual ledger  
3. OGEC occlusion gate + propagate path  
4. Exact-capture 入队改造 + stats  

### C2* 满配
1. HBG-RP `{g,p}` packetizer + clock gate  
2. ADP-MAC dual-side bit + donation  
3. ARM-Acc **与** MFBD（R2 主攻；两者都做：ARM 先、MFBD 紧随）  
4. SP-Gate（二期可并行，仍纳入全做）  
5. stats / 消融梯子  

### 明确不做（本阶段）
- 覆盖旧 C1/C2/TSBG 文件  
- 重跑 quarantine VCS  
- 把 Motion-XOR 写成 AEE 算法创新  

---

## 2. Codex 提速后的工期（修订）

原先 1 人手搓估计 C1* 9-13 + C2* 12-17 ≈ **21-30 人周**。  
**Codex 并行（规格清晰 + 人工 review）** 经验系数 roughly **0.35-0.5** 日历时间，但 review/合入仍要人盯。

| 阶段 | 内容 | 日历（1 人盯 Codex + Grok 出规格） | Codex 任务数 |
|---|---|---:|---:|
| P0 | ATLIF 契约 + ep34 profiling + 目录骨架 | 2-4 天 | 0-1 |
| P1 | C1* OP-STW + C2* HBG-RP（最小故事） | 4-7 天 | 2 |
| P2 | C1* PRRC+OGEC；C2* ADP-MAC | 7-12 天 | 2-3 |
| P3 | ARM-Acc + MFBD + SP-Gate | 7-12 天 | 2-3 |
| P4 | 统一消融 / VCS 新身份 / 论文表 | 5-8 天 | 1 |
| **合计日历** | 全做 R1+R2 | **约 3.5-6 周** | ~8-10 会话 |

并行度：C1* 与 C2* **两条 Codex 线**同时开，总日历取 max(C1,C2)+联调，而不是相加。

---

## 3. 双轨并行排期（推荐）

```text
Week 0.5  P0  契约+profiling+repo 骨架
Week 1    P1a C1* OP-STW          ||  P1b C2* HBG-RP
Week 2    P2a C1* PRRC            ||  P2b C2* ADP-MAC 骨架
Week 3    P2c C1* OGEC+propagate  ||  P2d C2* ADP donation
Week 4    P3a ARM-Acc             ||  P3b SP-Gate
Week 5    P3c MFBD                ||  P4  联调消融
Week 6    buffer / VCS 新 identity（另批）
```

---

## 4. 目录骨架（待批后 NEW only）

```text
SDformer/hw_autoresearch_nts07/
  rtl_c1star_grokbot/
    README_GROKBOT.md
    c1s_op_stw_predictor.sv
    c1s_prrc_ledger.sv
    c1s_ogec_gate.sv
    c1s_exact_capture_wrap.sv
    c1s_stats.sv
    c1s_top.sv
  rtl_c2star_grokbot/
    README_GROKBOT.md
    c2s_hbg_rp_packetizer.sv
    c2s_adp_mac.sv
    c2s_arm_acc.sv
    c2s_mfbd.sv
    c2s_sp_gate.sv
    c2s_stats.sv
    c2s_top.sv
  tb_c1star_grokbot/
  tb_c2star_grokbot/
  docs_grokbot/
    ATLIF_contract_r1.md
    ablation_ladder.md
```

每文件头必须含: `GROKBOT NEW FILE -- iscas_ssh / Codex`

---

## 5. Codex 任务卡（可直接粘贴）

### Card A — C1* OP-STW
- Goal: implement `c1s_op_stw_predictor.sv` + tiny TB; wake bitmap from |dF| and event_count; no modify existing C1.
- Accept: iverilog/VCS compile; wake rate printable; README lists ports.

### Card B — C2* HBG-RP
- Goal: `{g,p}` packetizer; gate clocks enable; parameterized eps and payload width.
- Accept: directed TB for amp=0 / small / large; no old TSBG edits.

### Card C — C1* PRRC+OGEC
- Depends on A. Residual budget + match/propagate paths + stats.

### Card D — C2* ADP-MAC
- Depends on B. Dual-side bit skip + optional donation; iso-interface to Acc24 stub.

### Card E — ARM-Acc
- Hyp lanes K=4/8; winner commit; TB hyp switch.

### Card F — MFBD + SP-Gate
- Motion-bundle broadcast; attention-mass cold suppress.

### Card G — Ablation harness
- Ladder scripts producing CSV for paper tables.

（完整端口表见 `05_C1star_C2star_microarch_sketches.md`）

---

## 6. 必须先钉死的 ATLIF 契约（阻塞 C2*）

请用户确认（或授权 Grok 给默认并写进 `ATLIF_contract_r1.md`）:

1. amp 位宽？建议 Q1.7 或 int8  
2. eps 默认？建议相对 max_amp 的 1/64 或绝对 1  
3. **不可吸收进下一层 W？**（是 → HBG-RP 成立；否 → 故事弱化）  
4. g 用硬阈值还是 soft？建议硬阈值门控  
5. 与现有 ep34 发放统计如何对齐？

---

## 7. Grok Bot vs Codex 分工

| 角色 | 负责 |
|---|---|
| **Grok Bot** | 规格、端口、消融梯子、进度、review、ismd-nemo 跑数、不覆盖纪律 |
| **Codex** | 按 Card 写 NEW SV/TB、本地/服务器仿真至绿 |
| **你** | 契约拍板、阶段性批准合入、最终 VCS 授权 |

---

## 8. 下一步（等一句批准）

1. 写 `ATLIF_contract_r1.md`（可用默认并标 TBD）  
2. 开 NEW 目录骨架 + README_GROKBOT  
3. 同时丢 Codex Card A + Card B  
4. 开 ep34 profiling 脚本（NEW）  

**不自动改旧文件；不自动开生产 VCS。**
