# 实现路线图（6 周 → DATE 2027 投稿）

## Week 1：规格冻结 + Golden 向量

| 任务 | 产出 | 状态 |
|------|------|------|
| 锁定 NTS-07b ep29 checkpoint | 路径写入 README | ✓ 完成 |
| Autoresearch 11 轮网格搜索 | `docs/10_autoresearch实验结果.md` | ✓ 完成 |
| 推荐配置终极组合 | `scripts/configs/best_config.json` | ✓ 完成 |
| `export_hw_golden.py` 从 overlay 导出 H60 中间值 | `tb/golden/*.json` | 待做 |
| RTL 仿真对齐 gate/attn | 容差报告 | 待做 |

## Week 2：H60 引擎完善

| 任务 | 产出 |
|------|------|
| `h60_attention_engine` 全流程时序 | 波形图 |
| `shiftmax_unit` bit-accurate 验证 | 与 Python 对比表 |
| Yosys 综合 H60 + Shiftmax | `reports/h60_area.json` |

## Week 3：Sparse MAC + 控制器

| 任务 | 产出 |
|------|------|
| `sparse_mac_pe` 8-lane 扩展 | 综合通过 |
| `nts07_controller` TTB FSM | 状态机图 |
| 接入 `hw/rtl/top.v` 或独立 `nts07_top` | 集成仿真 |

## Week 4：系统仿真 + Perf

| 任务 | 产出 |
|------|------|
| 单帧 end-to-end RTL+perf model | Mcycles 报告 |
| Autoresearch 扫 P0/P1 策略 | dashboard |
| 与 spike_profile 对齐 firing | 调度表 CSV |

## Week 5：DC 综合 + 面积功耗

| 任务 | 产出 |
|------|------|
| 28nm DC 综合 top | area/timing |
| 功耗估算（SAIF 或公式） | mW @ 30FPS |
| 对标表填 FireFly-T/ASNA-Flow | Table in paper |

## Week 6：论文硬件章节

| 任务 | 产出 |
|------|------|
| Architecture figure | PDF |
| Microarch H60 | 与 FireFly-T 并排 |
| Evaluation: area/power/latency | DATE 表格 |
| 开源 RTL tarball | Supplementary |

---

## 风险登记

| 风险 | 缓解 |
|------|------|
| NTS-08/09 替代 NTS-07b | 引擎接口不变，仅更新 μ/threshold 常数 |
| DRAM 带宽瓶颈 | Window 流式 + 权重预取 |
| Shiftmax 数值漂移 | golden 回归测试 |
| DATE 审稿要求 P&R | Week 5 后启动 place-only |

---

## 与软件实验并行

| 软件 | 硬件 |
|------|------|
| DSEC valid825 主表 | perf model 周期占比 |
| MVSEC 泛化 | 不改 RTL，仅验证精度 drift |
| NTS-09 训练 | 更新 `MU_Q8`/threshold 寄存器默认值 |