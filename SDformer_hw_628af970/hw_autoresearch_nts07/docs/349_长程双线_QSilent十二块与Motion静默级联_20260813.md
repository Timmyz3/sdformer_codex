# 长程双线：Q-silent 十二块外推与 Motion 静默级联

> 日期：2026-08-13  
> 范围：接 `docs/348` 后继续，不暂停确认  
> 原则：调度模型与净表分开；空行跳过按真实占用边带计周期

## 1. 本轮做了什么

### Local5：把 Q-silent 接到 12-block 账本

用已封存的 100-group Q-silent/residual TCFM5 L1 周期，按 encoder descriptor 的 stage 均值外推：

| 口径 | residual | Q-silent | 加速 |
|---|---:|---:|---:|
| head-window 21600 | 16,434,547 | 9,099,342 | **1.806x** |
| scheduler window-group 1320 | 2,827,479 | 1,509,620 | **1.873x** |

证据：`results/local5_qsilent_12block_frame_20260813`  
这是 `[rtl校准模型]`，不是 21600-group RTL，也不含 ATLIF/IO。

### Motion：先测共享后端上界，再做空行精确跳过

RQTB2S ready=1 相位拆分（138 行 RTL）：

- 每行 **build 恒为 226**（225 pair + 1）
- emit 从 2 到 1163，总 emit 占 63.7%
- 共享 encoder/Shiftmax、双 workspace 的合法调度：按 block **1.292x**（85912→66487）

证据：`results/h67_laws_shared_backend_phase_20260813`  
这是相位 RTL + 调度模型，还不是双 directory 网表。

空 K 行有 33/138。它们仍在付 226 拍扫描。旁路 `h67_empty_row_skip_2s` 用写 K 时留下的 1-bit occupancy 直接跳过：

| 项 | 数 |
|---|---:|
| 空行 | 33 |
| 跳过拍 | 66（约 2 拍/行） |
| 密行 | 105，78,380 cycle |
| 合计 | 78,446 |
| 相对相位顺序 | **1.095x** |

功能通过 Icarus+Verilator+Yosys。单独看空行跳过只有约 9.5%，不能当主贡献。

空行跳过后再做同一套共享后端调度，上界是 **1.454x**（85912→59088）。仍标为调度上界，不是已布线双核。

## 2. 现在能写的硬件贡献

### Local5（本轮加强）

1. 五邻域 inverse-stencil + 五色投影  
2. **Query-Silent exact cascade**（部署切片 1.70x，12-block 外推约 1.81–1.87x）

### Motion

1. 可逆时间 score-class 合并（ep35 LFSR 公平包 1.1865x）  
2. **Query/K 静默级联是同一家族，但 Motion 侧空行跳过只有 1.10x，只作支撑**  
3. 共享后端 row-pipeline 是 1.29–1.45x 上界，晋级条件是双 directory/K 净表

## 3. 明确不写

- 整核双复制 1.87x（ANT≈0.94，已否决）
- 把 1.45x 写成已实现 dual-workspace 周期
- 把 12-block 1.81x 写成 21600-group RTL
- 空行跳过单独当 DATE 主创新

## 4. DATE 复审（本轮结束后的整包）

| 对象 | 分 | 说明 |
|---|---:|---|
| Local5 证据包 | 4.2 / 5 | 前端瓶颈被打穿，并接到 12-block 账本 |
| Motion 新机制 | 3.2 / 5 | 相位拆分清楚，空行跳过太小，pipeline 未净表化 |
| 整篇 DATE | **3.4 / 5 Borderline** | 比 348 略升，仍缺 DC/SAIF 与 encoder 分账 |

## 5. 下一档（无需确认即可继续）

1. Motion 双 directory/K 净表，把 1.29x 从调度模型升成 `[rtl]`  
2. Local5 用 Q-silent tile 跑至少一个真实 descriptor window 的 scheduler+score 同顶层  
3. DC 机到位后做 Q-silent 开/关同 SDC 对照
