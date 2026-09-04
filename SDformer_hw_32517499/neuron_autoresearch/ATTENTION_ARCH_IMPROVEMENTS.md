# SC Attention — 全景架构改进方向

## 1. 窗口设计（改动窗口形状/大小）

### 1a — 拉长时间窗口
- **现状**: window=[2,9,9]，时间仅 2 步
- **改进**: window=[5,5,5] 或 [10,5,5]，给注意力更多时间上下文
- **硬件影响**: 窗口 token 数从 162 变 125 或 250，仍在 SRAM 可行范围
- **风险**: 中等。PSN 的时间步是顺序的，加大窗口可能改变 BN 行为

### 1b — 非对称窗口（空间扁窗口）
- **动机**: 光流的运动在 x/y 方向不对称（车辆水平运动 > 垂直运动）
- **改进**: window=[2,12,6]（宽窗口，匹配水平运动）
- **硬件**: token 数不变（144），零额外成本

### 1c — Event-Adaptive Windowing（事件密度自适应窗口）
- **空白**: 文献中完全没有 SNN + 事件自适应窗口
- **想法**: 根据窗口内事件密度动态调整窗口大小
  - 高密度区域 → 大窗口（捕捉更多上下文）
  - 低密度区域 → 小窗口（节省计算）
- **实现**: 在 ATLIF 上加一个 event_density_counter → 控制 window_size
- **论文价值**: OPEN RESEARCH GAP，高影响力

## 2. 跨窗通信（窗口间交互）

### 2a — Temporal Cross-Window Attention
- **现状**: 只有 SW-MSA 做空间跨窗，时间维没有跨窗交互
- **改进**: 加一层 temporal cross-window：取相邻时间步的窗口做 attention
- **动机**: 光流本质上需要前后帧对比，2 步窗口 + 跨步通信才有效
- **硬件**: 额外一次 attention，约 +20% 计算

### 2b — ConvLSTM Bridge（借鉴 Video Swin-CLSTM）
- **改进**: 在 SW-MSA 和 MLP 之间加 ConvLSTM
- **动机**: 弥补 window 割裂带来的时序记忆缺失
- **硬件**: ConvLSTM 是标准模块，有成熟硬件实现

## 3. 多尺度注意力（不同粒度）

### 3a — Dual-Path Attention（粗粒度 + 细粒度）
- **现状**: 每个 stage 固定一个 window_size
- **改进**: 同一 stage 内跑两条 path：
  - Path A: window=[2,4,4]（细粒度，局部运动）
  - Path B: window=[2,9,9]（粗粒度，全局运动）
  - 合并 gate = α × gate_A + (1-α) × gate_B
- **硬件**: 额外 50% 计算（细粒度 path 窗口小，token 多）

### 3b — Stage-Adaptive Window（不同 stage 不同窗）
- Stage 0 (大图 144×192): window=[2,12,12]（更大窗捕捉大运动）
- Stage 3 (小图 18×24): window=[2,6,6]（小图用小窗）
- **零代码改动**：只改 config 的 window_size

## 4. SC 分数计算改进（不改窗口结构）

### 4a — SC + Temporal Consistency（时间一致性得分）
- **动机**: 光流在相邻时间步应该一致
- **改进**: score = SC_token + λ × |gate(t) - gate(t-1)|（惩罚时间不连续性）
- **代码**: 在 `_signed_consensus_token_scores` 加时间差分项

### 4b — SC + Motion Magnitude Weighting（运动幅度加权）
- **动机**: 运动大的 token 更重要
- **改进**: 对变化大的 token 给更高 attention weight
- **实现**: 用相邻帧的 voxel diff 计算 motion_magnitude → 乘到 gate 上

### 4c — SC + Directional Channels（方向通道）
- **动机**: 光流有 x/y 两个方向，当前 SC 不分方向
- **改进**: 把 Q/K 按 x/y 分开计算 SC，然后合并
- **硬件**: token-popcount × 2（x 方向 + y 方向），硬件加倍但仍然是纯 shift

## 5. 硬件协同设计

### 5a — Window-Strided SRAM Pipeline
- **现状**: 全帧处理，激活值 149MB（stage 3 decoder）
- **改进**: stripe-based 流式处理：一次只加载一个 window strip
- **硬件增益**: 片上 SRAM 从 512KB → 38KB/window，适配 28nm SRAM

### 5b — Event-Driven Window Activation
- **改进**: 空 window（无事件）直接跳过整个 attention block
- **实现**: 在 event_voxel 上加一个 window_valid mask
- **硬件增益**: 30-50% window 可能为空（取决于场景），直接省掉那些计算

## 执行优先级

| 优先级 | 方向 | 改动量 | 论文价值 | 风险 |
|--------|------|--------|---------|------|
| **P0** | 1a 拉长时间窗口 | 改 config | 中 | 低 |
| **P0** | 3b Stage-Adaptive Window | 改 config | 中 | 低 |
| **P1** | 4b Motion Weighting | ~50 行 | 中高 | 低 |
| **P1** | 1c Event-Adaptive Windowing | ~200 行 | 🏆 极高 | 中 |
| **P2** | 5b Event-Driven Window Skip | ~100 行 | 高 | 低 |
| **P2** | 2a Temporal Cross-Window | ~200 行 | 高 | 中 |
| **P3** | 3a Dual-Path Attention | ~300 行 | 高 | 中 |
