# M722R2 作者交接：三行 LB-FUSE 终止，不开 RTL

结论是 `KILL_NO_RTL`。在同样的 96 lane、K8、240 KiB、Acc24、dense commit 和单 1RW psum 端口下，公平 A1-OSG 同样采用三行生命周期，并用 D3 合法宽度条带把片外 psum spill 降到 0。LB-FUSE 因按 source 直接发射而失去 destination packing，不能用“避免 RMW/DRAM”取得优势。

## 冻结结果

| 口径（D0+D2+D3，三序列 S3x10） | A1-OSG | LB-FUSE | 判定 |
|---|---:|---:|---|
| 周期 | 21,590,945,350 | 23,377,337,337 | A1/LB = 0.923584×，LB 慢 8.27% |
| group | 827,946,728 | 1,170,190,821 | LB 为 1.413365× |
| 片上 psum RMW | 476,897,315,328 B | 549,335,071,872 B | LB 多 15.19% |
| 片外 psum spill | 0 | 0 | 没有流量收益 |
| dense commit | 11,612,160,000 B | 11,612,160,000 B | 相同 |
| Acc24 数值 mismatch | 0 | 0 | 1200 plane 精确 |

三条序列 A1/LB 分别为 0.926111×、0.923851×、0.920822×，没有依赖某一场景翻转。

## 第一性原理裁决

stride-2 K3 ConvTranspose 的一个 destination 最多接收四个不同 source position 的贡献。A1-OSG 以 destination 为键打包这些贡献；LB-FUSE 按 source 顺序直接发射，天然无法跨 source position 合并。若给 LB-FUSE 增加 destination-keyed context、descriptor 和完成状态来恢复这部分 packing，它会重新收敛到 A1-OSG/PIDP，而不是一个新的低成本机制。

三行 line buffer、polyphase/deconvolution decomposition 和 Acc16 缩窄均已有直接先例。H67 的对象差只包括 binary ATLIF descriptor、1/2/2/4 asymmetric taps、signed INT8/Acc24、240 KiB 和 96-wide issue，不能把经典 line buffer 本身包装成 novelty。

## 精度与容量

D3 完整 96-channel Acc16 三行存储总计 206,336 B，且完整 S3x10 trace 的任意顺序绝对前缀界为 7,288，满足 Acc16。D0 的对应界为 62,696，不满足 Acc16；所有层均满足 Acc24。公平 A1 的 D3 Acc24 采用 256+64 宽两条带，只重复一个 input column，总计 243,200 B，低于 240 KiB 且没有片外 spill。

D3 的 Acc16 方案可作为普通 precision/storage 消融，但不是 line-buffer 加速创新；D3 `Cout48×2` Acc24 反而只有 0.735360× A1/LB，因此没有准入。

## 身份与复算

- Canonical result：`results/m722r2_lb_fuse_decoder_cpu_fastkill_r1_20260828`
- Contract：`contracts/m722r2_lb_fuse_decoder_cpu_fastkill_contract_r1_20260828.json`
- 作者复算：`python3 reviews/m722r2_lb_fuse_decoder_cpu_fastkill_author_handoff_r1_20260828/recompute_handoff.py`
- `docs/359` SHA 保持 `dedde7ce...`。

R1 在任何 numeric replay 之前因 preflight 错误退出，且没有 canonical result。R2 只修复根 seal 与嵌套 seal 的成员判定，没有修改算术、周期、存储或 gate。当前仍要求新的独立 result hammer 后才能引用；本交接不授权 RTL、VCS、EDA、GPU、远端任务、系统加速或论文 headline。
