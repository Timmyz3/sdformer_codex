# 算法侧新算子合同候选队列（通往 4.0 的唯一正道）

日期：2026-08-18。来源：两轮硬件创新攻击的收敛结论（CLAUDE_INNOVATION_ATTACK_ROUND2_MOTION/LOCAL5）。用户已批准加入队列。

## 背景

两轮独立攻击（4 个 opus agent）一致结论：**硬件侧拆数据流的上限是创新 3.5-3.7**（Motion quotient-file 侧车三腿已过实测数据；Local5 C1 分数统计跨 pair 边界保持）。到 4.0 必须满足 docs/433：**新算法算子合同 + 改硬件存储/执行对象**。合同在先，硬件引擎随后。

## 候选队列（按优先级）

### P0-1：Motion T>2 时间窗合同

- 现网：window (2,15,15)，T=2 时间对（RQTB 的"时间 slot → 可逆 Q7 商 + multiplicity"依赖 pair 结构）
- 候选：window (4,15,15) 或 (5,15,15)——时间商从 pair 变成四元组/五元组。时间维的商结构、mask 语义、normalization 对象全部变，是新算子合同
- 关键问题：T=4 的训练稳定性（SNN 时间步语义）、spikes 预算影响、与 Motion-XOR 的兼容
- 验证路径：先小配置（short 训练）验证 loss 不塌 → fullres ft → valid825 对比 ep35 锚点

### P0-2：Motion 跨窗语义合同

- 现网：SWIN 非重叠 tile，窗口间无共享（这是硬件侧推翻"跨窗 quotient 持久"的原因）
- 候选：**带重叠的滑动窗 attention**（shift-window 或重叠一半）——让"相邻窗口共享 token"成为合同的一部分。这样硬件侧的跨窗 quotient 目录（J=0.84 数据）才有机制基板
- 风险：改变了 Swin 架构本身（与 baseline 论文对比口径要重新建立）；计算量增加
- 验证路径：overlap 窗配置 → 训练 → valid825 对比；硬件侧同步评估 quotient 目录的 bit 节省

### P1：Local5 C1 配套合同确认

- Local5 的 C1（分数统计跨 pair 边界保持）不需要新算法合同，但需要 rank-1 带 Q 标签 dump 裁决统计平面的 exact 身份（GPU 任务，排队等缺口审计 agent 让卡）
- dump 规格：Local5 ep44 全 valid825 或子集，每 pair 输出 (q1[p], k1[p], q1[p+1]) 的三元统计一致率

### P2：H86 存档参考

- H86（窗内 member-delta）已停线存档。其"目录差分"思想可作跨窗语义合同的参考，但轴必须改成 T450 扫描名单级（硬件侧 docs/442 的否决理由）

## 执行约定

- 新合同训练必须 seed 0、不混 MDR/MVSEC/transfer 表、short 验证先过再 fullres
- 每个候选先写合同草案（新算子合同一句话 + 新存储对象 + 新执行对象）再开训练
- GPU 队列纪律：一次一个任务，与 DATE 缺口审计 agent 协调

## 状态更新（2026-08-19 凌晨）

- 三份合同草案已完成：CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md（D1 Motion T>2 时间商 / D2 跨窗语义 / D3 A3S 各向异性 stencil）
- 排序：**D1 > D3 > D2**。D1 是唯一被 round2 攻击点名"可证伪 4.0"的类别；训练可证伪标准：seed 0 short loss 不塌 → fullres valid825 对比锚点（Motion 1.3297 / Local5 1.2819，±1% 通过线）
- **D1/D2/D3 全部实现完毕**（纯追加 mode h87/h88/h89 + CPU 单测 + launcher）
- **D1 训练干预**：lr 1e-4 发散 → 5e-5 ep1 +33% 越过失败线 → **lr 2.5e-5 重启（ep0 val 1.152 健康）**；位账重定价 −63.7% 全确定性 + 双峰可证结构（变体搜索裁决：保留 T=5）
- **D2 降级 side-note**（新颖性风险高：online-softmax 系 + 卷积 overlap 复用双线拥挤、净 exp 流量仅 −4.8%）；升格条件见 D2_NOVELTY_DOSSIER_20260819.md
- GPU 队列：D1 short（运行中）→ D3 short → Mode B 精度评估 → Local5 C1 预检
