# GrokBot native verdict (2026-09-11)

审的是隔离树 `sdformer_c1c2star_grokbot` 的卡片与 RTL，不是 245 张文献卡，也不是独立开题。独立自评 ~3.2/10 borderline 已经偏松。

## 实际落地了什么

可综合叶子 + iverilog 定向 TB，不是同资源加速，也不是 DSEC AEE。

字面电路：

| 模块 | 打开 SV 后的行为 |
|---|---|
| OP-STW | `wake = (\|flow_cur-flow_prev\| > TH) \|\| (event_cnt > TH)` |
| HBG-RP | `g = amp_valid && (\|amp\| > EPS); p = g ? amp : 0; pe_clk_en = g`（死区/时钟门，~24 cells） |
| OGEC | `exact_en = match_ok` 透传 |
| PRRC | 三级递减计数；`allow_exact = (budget != 0)`。草图里的整数 warp/ROI **没有** |
| exact_capture | `gated = allow_exact ? exact_en : 0` + 容量 FSM |
| ECP / MW / STH / SP | 阈值比较器 |
| TDE3-Prior | 1D 邻域 age 谁更年轻 → dir/conf |
| TMA-Agg | 1D 左右邻 feat 相等则一致性++ |
| BiSAT-Agg | `fwd == bwd` 则平均，不等则右移；无 SATMA |
| BUI-GuardSDSA | ub/lb 宽度 + MSB 幅度；无 bit-serial 回合 |
| CFP-ConfGate | `conf = sat(corr_peak)`；高则 propagate。TH_LO 死参 |
| SCI-CleanExit | `g = 255 - \|f1-f2_warp\|`；`f_hat` 未用；warp 由 TB 预供 |
| SMAM-RP | 有门则 payload 透传，Mask-Add 只 +1 |
| ADP-MAC | `do_mac = mac_en && !skip_a && !skip_b`；`FORBID_REORDER` 是注释参数 |
| Motion-TTB | 扫描 wake 打包 |
| MFBD | 只用 `bundle[0].hyp_id`；tile/dt 显式未用 |
| ARM-Acc | `acc[hyp_sel] += data` |
| wake_merge | 三路 OR |

`c1s_front_pipe` 只串 OP-STW ∥ MW → ECP。`c2s_back_pipe` 只串 HBG → SMAM，STH 并行。Card G/H（TDE3 / wake_merge / CFP / SCI / TMA / BiSAT / BUI）**没有**进 `c1s_top` / `c2s_top`。消融梯子是 N_TILE=8 的定向向量：ALWAYS wake_pop=64 → OPSTW=32；EXACT 把 exact_hit 49→19 是 `INIT_BUDGET=3` 的算术封顶。OpenROAD = 布线完整度。

## 仍是口号的部分

- 三刀信件：光流调度精确工作 / 双轨不可吸收 int8 / 匹配∩预算精确入队。刀1是两个比较器；刀2是死区；刀3是使能相与。
- first-HW TMA / BAT / CFP / SCI / BUI / TDE。近亲论文在 GPU/FPGA/LLM，这里是同名阈值玩具。
- ATLIF int8 payload 已是部署身份。合同自己写 DEFAULT DRAFT；ep35 profile 93/93 binary。本刊身份是**连续 θg**，不是二值发放，也不是另开 int8 包。
- PRRC 金字塔残差窗、OGEC 遮挡检测、MW motion-warp、SCI 自清洁迭代、BiSAT 反向相关、BUI 逐 bit 护栏。
- 18 模块动物园 = 18 个创新；路由 µm² / mW / 整网 FPS。
- Card I Top5（ResHTR / EDC / WinTok / AdjEvt / TID）与 EvQ / PredExit / BitHyp：只有草图。

## 还能当 A / B / X 的刀

**A（可抄先验，不得当本贡献）：** TDE-3、SciFlow、EMD-Flow CFP、BAT、PADE BUI-GF、SpAtten、SMAM Mask-Add、EDFLOW 闭环、FireFly-T AND-popcount。用来当对照和 related work。

**B（本网未解决洞）：** 真洞在 patch r1 可学习 T10 lifting + 完整常量编译 + 真实双消费者，精度门 AEE≤1.259、同资源服务≥15%。GrokBot 树**没有**对这个 B 作证。反向证据：Codex 已测残差卷积因果运动差分相对 bit-skip 是额外功，这直接压 OP-STW / MW-ΔBuf。

**X（相对 A 的真增量）：** **无标题级 X。** 阈值比较器 novelty 记 0–1。唯一不丢人的句子级约束是「连续 θg 不要吸进下一层 W」——这是身份纪律，不是 HBG-RP 模块。PRRC 账本 ∩ 容量 FSM 可留作**控制胶水**，不是刀3。

## 停什么

停布局，不杀家族。

1. **停三刀信件。** OP-STW / HBG-RP / OGEC×PRRC 不得再当 TCAS-II 标题。
2. **停 Card G/H 当增强主刀。** TDE3/TMA/CFP/SCI/BiSAT/BUI 全部降为对照或停实现叙事。
3. **不要开 Card I。** ResHTR / EDC / WinTok / AdjEvt / TID 停。
4. **停 grok46 二值岛。** MX3P、temporal-peer dirty-lane、LoAS-FTP 织物、mixed-horizon binary ATLIF、Bishop-ECP-bound：本刊 ATLIF 保持连续 θg。
5. **停模块动物园与 OpenROAD-as-PPA。** iverilog PASS ≠ 同资源加速。
6. **可留的胶水（非标题）：** wake_merge、PRRC 计数器、exact_capture 容量 FSM、Motion-TTB 打包、普通质量门。接上真实 T10 / 双消费者贵段之前，不要扩 RTL 树。
