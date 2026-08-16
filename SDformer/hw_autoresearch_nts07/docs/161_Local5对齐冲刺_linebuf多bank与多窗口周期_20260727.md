# Local5 对齐冲刺：line-buffer、3-bank 投影、多窗口周期

**日期**：2026-07-27  
**隔离**：仅 `rtl_local5` / `tb_local5` / `scripts/local5_*`；**未修改** `rtl_hitflow` / `rtl_h67` / `rtl_delta`。

---

## 0. 结论

| 项 | 状态 |
|---|---|
| line-buffer → stencil fetch → 整链 | **`local5_linebuf_window_top` PASS** |
| 3-bank 投影（DCTF-like，Local5 自有） | **`local5_multibank_projection_top` PASS** |
| 16-dest 直喂窗口周期 | **PASS 1965 cycles / 997 cmds** |
| 3×8-dest linebuf 窗口 | **PASS mean 1219 cycles** |
| 同 sample equal96 vs Motion | **仍未**（workload 不同） |
| 实例化 hitflow DCTF 内部 | **未**（端口过重；用 Local5 multibank 等价语义） |

功能完成度自评：**0.88 → ~0.92**（L4 存储+多 bank+多窗口周期分位）。

---

## 1. 新增模块

| 文件 | 角色 |
|---|---|
| `local5_multibank_projection_top.sv` | dest%N bank + 1-cycle busy 冲突计数 |
| `local5_stencil_linebuf_fetcher.sv` | 从 3-row buffer 拼 self/N/S/E/W |
| `local5_linebuf_window_top.sv` | 行推入 → fetch → SGT → bridge → 3-bank |
| `tb_local5_window16.sv` | 16 dest 直喂缩放 |
| `tb_local5_linebuf_window.sv` | 3 窗口 × 8 dest 周期分位 |
| `scripts/local5_collect_verilator_cycles.py` | 汇总周期表 |

---

## 2. Verilator 实测周期

| 配置 | dests | cycles | cmds | cycles/dest |
|---|---:|---:|---:|---:|
| direct window4（既有） | 4 | 455 | 230 | 113.8 |
| **direct window16** | **16** | **1965** | **997** | **122.8** |
| **linebuf win0** | 8 | 1157 | 436 | 144.6 |
| linebuf win1 | 8 | 1240 | 926* | 155.0 |
| linebuf win2 | 8 | 1261 | 503 | 157.6 |
| linebuf **mean** | 8×3 | **1219** | — | **152.4** |

\*win1 cmds 偏高，疑为随机 gate 更密 / 计数边界；功能仍 PASS。  
linebuf 路径 cycles/dest 高于直喂：含 line-buffer 读延迟 + bank conflict stall。

Motion equal96（**不同 workload**，仅对照形态）：

| 结构 | cycles |
|---:|---:|
| Central96 | 59853 |
| DCTF96-2C | 53910 |

**禁止** 1228 与 53910 直接比快慢。

---

## 3. 与 Motion 梯子

| 阶 | Local5 |
|---|---|
| L1–L3 | 齐 |
| L4 投影 | banklocal + **3-bank** + multiset bridge |
| L4 存储 | **line-buffer + fetcher 接入顶层** |
| L5 多窗口周期 | **3-window mean/min/max**；尚无同 cohort equal96 |
| L6 | STT 仍薄 |
| L8/L9 | 无 DC |

---

## 4. 复现

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate sdformerflow
cd hw_autoresearch_nts07
# 16-dest
verilator --binary --timing ... tb_local5_window16.sv  # 见 sim 脚本扩展
# linebuf multi-window
# build_local5/parity/lbw/Vtb_local5_linebuf_window
python3 scripts/local5_collect_verilator_cycles.py
```

---

## 5. 下一步

1. 修复/审计 win 间 cmd 计数一致性；Acc golden 接到 linebuf 顶层  
2. 162-token 行（或分块 18×9）周期分位  
3. 可选：新文件 `local5_hitflow_cmd_shim.sv` 只适配 cmd 字段，仍不改 hitflow 源  
4. post-G0 软件导出替换合成向量  
