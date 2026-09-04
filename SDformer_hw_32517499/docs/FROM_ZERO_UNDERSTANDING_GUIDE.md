# SDFormer Baseline 从零理解指南

**作者**: 小虾 🦐  
**日期**: 2026-03-27  
**目标读者**: 小朱（有基本深度学习知识和简单Python理解）

---

## 📚 目录

1. [整体架构理解](#1-整体架构理解)
2. [数据流详解](#2-数据流详解)
3. [核心模块详解](#3-核心模块详解)
4. [代码入口分析](#4-代码入口分析)
5. [改进接口标注](#5-改进接口标注)
6. [软硬件协同优化建议](#6-软硬件协同优化建议)

---

## 1. 整体架构理解

### 1.1 这是什么模型？

SDFormer是一个**事件相机光流估计**模型，核心特点是：

- **输入**：事件相机数据（不是普通图像）
- **输出**：光流（每个像素的位移向量）
- **架构**：SNN（脉冲神经网络）+ Swin Transformer + UNet

### 1.2 事件相机 vs 普通相机

| 特性 | 普通相机 | 事件相机 |
|------|---------|---------|
| 输出 | 完整图像帧 | 异步事件流 |
| 频率 | 固定（如30fps） | 高动态（可达1MHz） |
| 数据格式 | `[B, 3, H, W]` | `[B, T, 2, H, W]` |
| 特点 | 有运动模糊 | 无模糊、低延迟 |

**事件数据解释**：
- `B` = Batch size
- `T` = 时间bins（如10个时间切片）
- `2` = 正负极性（光变亮/变暗）
- `H, W` = 高度和宽度

### 1.3 为什么用SNN？

事件相机的输出本身就是"脉冲"形式（事件触发），天然适合用脉冲神经网络处理：

1. **事件驱动计算**：只在有事件时计算，省功耗
2. **时间信息保留**：SNN天然处理时序数据
3. **硬件友好**：脉冲是二值信号，适合FPGA/ASIC实现

---

## 2. 数据流详解

### 2.1 输入数据

```python
# 从数据加载器出来的数据
batch["event_voxel"]  # shape: [B, 10, H, W] 或 [B, 10, 2, H, W]
batch["gt_flow"]      # shape: [B, 2, H, W]（ground truth光流）
batch["valid_mask"]   # shape: [B, H, W]（有效区域掩码）
```

**10个bins是什么？**
- 把一段时间内的事件按时间分成10个切片
- 每个bin内累计该时间段的事件

### 2.2 预处理阶段

```python
# 在 backbone.py 的 _preprocess_input 方法中

# Step 1: 正负极性分离
if event_voxel.dim() == 4:  # [B, T, H, W]
    pos = torch.relu(event_voxel)   # 正事件
    neg = torch.relu(-event_voxel)  # 负事件
    chunk = torch.stack((pos, neg), dim=2)  # [B, T, 2, H, W]

# Step 2: Spike编码
chunk = self.spike_encoder(chunk)  # 转换为脉冲表示

# Step 3: 归一化
chunk = self._normalize_nonzero(chunk)  # min-max归一化
```

**输出shape变化**：
```
输入: [B, 10, H, W]
     ↓ 预处理
输出: [B, 10, 2, H, W]
```

### 2.3 Patch Embed阶段

这是模型的第一层，把事件数据转换成token表示：

```python
# 位置: third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py
# 类: MS_PED_Spiking_PatchEmbed_Conv_sfn

# 内部流程：
输入: [B, 10, 2, H, W]
    ↓ 时间维度重排
[B, T=10, C=2*base_channels, H, W]
    ↓ head卷积
    ↓ conv下采样
    ↓ residual块 × 2
    ↓ proj投影
输出: [T, B, C, H/4, W/4]  # 注意：时间维度放在前面！
```

**关键点**：
- 这不是简单的patch切分，而是包含了一个小型前端网络
- 输出把时间维度`T`放在第一位（这是SNN的常见格式）

### 2.4 Encoder阶段

模型有4个encoder stage，每个stage包含：
1. 多个Swin Transformer Block
2. 一个下采样层（最后一层除外）

```python
# 配置中的定义
swin_depths = [2, 2, 6, 2]      # 每个stage的block数
swin_num_heads = [3, 6, 12, 24] # 每个stage的注意力头数
window_size = [2, 9, 9]          # 局部窗口大小 [T, H, W]
```

**每个stage的shape变化**：
```
Stage 1: [T, B, 96, H/4, W/4]   → [T, B, 192, H/8, W/8]
Stage 2: [T, B, 192, H/8, W/8]  → [T, B, 384, H/16, W/16]
Stage 3: [T, B, 384, H/16, W/16] → [T, B, 768, H/32, W/32]
Stage 4: [T, B, 768, H/32, W/32] → [T, B, 1536, H/64, W/64]
```

### 2.5 Decoder阶段

UNet风格的解码器，逐步上采样并融合encoder的skip连接：

```python
# 每层decoder输入包括：
# 1. 当前decoder的特征
# 2. 对应encoder的skip连接
# 3. 上一层预测的光流

# 输入通道数 = 2 * input_size + prediction_channels
```

**Skip连接的作用**：
- 保留encoder不同尺度的细节信息
- 类似UNet的跳跃连接

### 2.6 Flow Head阶段

每个decoder层后面都接一个flow head，预测该尺度的光流：

```python
# 位置: Spiking_modules.py - MS_SpikingPredLayer

# 流程：
输入: decoder特征
    ↓ Spiking_neuron（脉冲神经元）
    ↓ 1x1 Conv2d
输出: [T, B, 2, H, W]  # 带时间维的光流
```

### 2.7 时间聚合阶段

最后，把带时间维的光流聚合成最终光流：

```python
# 位置: Spiking_STSwinNet.py - forward方法

flow = torch.sum(flow, dim=0)  # [T, B, 2, H, W] → [B, 2, H, W]
```

**为什么求和？**
- SNN每个时间步输出一个脉冲响应
- 累积所有时间步的响应得到最终预测

---

## 3. 核心模块详解

### 3.1 Spiking Neuron（脉冲神经元）

**位置**: `src/models/sdformer/spiking_neurons.py`

脉冲神经元是SNN的基本单元，模拟生物神经元：
- 接收输入脉冲
- 累积膜电位
- 当电位超过阈值时发放脉冲
- 电位重置

```python
# 关键参数（在配置中）
neuron:
  type: psn           # Parametric Spiking Neuron
  v_th: 0.1           # 阈值
  v_reset: null       # 重置值（null=软重置）
  tau: 2.0            # 时间常数
  surrogate_fun: surrogate.ATan()  # 梯度替代函数
```

**可改进点**：
- 尝试不同neuron类型（LIF, IF, PLIF）
- 调整阈值和tau
- 添加自适应阈值

### 3.2 Attention（注意力机制）

**位置**: `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py`

当前使用的是**Spiking Window Attention**，不是标准的softmax attention：

```python
# 核心流程（简化版）
def forward(q, k, v):
    # 1. 生成q（查询）
    q = self.q_proj(x)
    
    # 2. 生成带位置编码的k（键）
    k = self.k_proj(x) + pos_embed
    
    # 3. 用q的聚合token调制k
    attn = q * k  # 这不是标准QK^T！
    
    # 4. 投影回来
    out = self.proj(attn)
    return out
```

**关键区别**：
- 标准Transformer: `Attention = softmax(QK^T/√d)V`
- Spiking版本: `Attention = Q ⊙ K`（element-wise乘法）

**可改进点**：
- 添加稀疏性（只计算重要token）
- 实现硬件友好的window attention
- 尝试不同的attention变体

### 3.3 Swin Block

**位置**: `Spiking_swin_transformer3D.py - MS_Spiking_SwinTransformerBlock3D`

每个block包含：
1. Attention（注意力）
2. MLP（前馈网络）
3. 残差连接

```python
def forward(x):
    # 1. Attention + 残差
    x = x + self.attn(self.norm1(x))
    
    # 2. MLP + 残差
    x = x + self.mlp(self.norm2(x))
    
    return x
```

**MS版本的特点**：
- 脉冲神经元在卷积之前
- 残差连接的位置不同于原始SEW版本

### 3.4 Token Pruning（Token剪枝）

**位置**: `src/models/modules/sparse_ops/token_pruning.py`

已实现的稀疏化方法：

```python
class StructuredTokenPruner:
    """按活动强度保留top-k token"""
    def forward(x):
        # 1. 计算每个token的活动强度
        activity = x.abs().mean(dim=channel)
        
        # 2. 选择top-k
        topk_indices = torch.topk(activity, k=keep)
        
        # 3. 生成mask
        mask = torch.zeros_like(activity)
        mask[topk_indices] = 1
        
        # 4. 应用mask
        return x * mask
```

**可改进点**：
- 添加硬件感知的剪枝（如按window剪枝）
- 实现动态剪枝阈值
- 添加剪枝恢复机制

---

## 4. 代码入口分析

### 4.1 训练入口

**文件**: `src/trainers/train.py`

```python
def main():
    # 1. 加载配置
    cfg = load_config(args.config)
    
    # 2. 构建数据集
    train_dataset = build_dataset(cfg, "train")
    
    # 3. 构建模型
    model = build_model(cfg).to(device)
    
    # 4. 训练循环
    for epoch in range(epochs):
        for batch in train_loader:
            outputs = model(batch)
            loss = loss_fn(outputs, batch["gt_flow"])
            loss.backward()
            optimizer.step()
```

### 4.2 模型构建

**文件**: `src/models/sdformer/backbone.py`

```python
class SDFormerFlowAdapter(nn.Module):
    def __init__(self, cfg):
        # 1. 加载上游模型
        self.model = MS_SpikingformerFlowNet_en4(...)
        
        # 2. 添加spike encoder
        self.spike_encoder = build_module("spike_encoding", ...)
        
        # 3. 添加预处理模块
        self.preprocess_modules = nn.ModuleList([...])
    
    def forward(self, batch):
        # 1. 预处理输入
        pre = self._preprocess_input(batch["event_voxel"])
        
        # 2. 模型推理
        output = self.model(pre["chunk"])
        
        # 3. 返回结果
        return {"flow_pred": output["flow"][-1], "aux": {...}}
```

### 4.3 数据加载

**文件**: `src/datasets/optical_flow_dsec.py`

```python
class DSECDataset:
    def __getitem__(self, idx):
        # 加载事件数据
        event_tensor = np.load(event_path)  # [T, H, W]
        
        # 加载光流真值
        gt_flow = np.load(flow_path)  # [2, H, W]
        
        return {
            "event_voxel": torch.from_numpy(event_tensor),
            "gt_flow": torch.from_numpy(gt_flow),
            "valid_mask": torch.from_numpy(mask)
        }
```

---

## 5. 改进接口标注

### 5.1 五个关键改进接口

| 接口 | 文件位置 | 改进方向 |
|------|---------|---------|
| Patch Embed | `Spiking_modules.py` | 输入编码优化 |
| Attention | `Spiking_swin_transformer3D.py` | 注意力机制改进 |
| Encoder Block | `Spiking_swin_transformer3D.py` | 模块增减 |
| Decoder | `Spiking_modules.py` | 解码优化 |
| Spiking Neuron | `spiking_neurons.py` | 神经元类型替换 |

### 5.2 配置驱动的设计

项目支持通过配置文件启用不同模块：

```yaml
# configs/sdformer_baseline.yaml

model:
  # 注意力配置
  attention:
    type: baseline  # 可改为 window_spike
  
  # 稀疏化配置
  sparsity:
    enabled: true
    token_keep_ratio: 0.7
    window_enabled: true
    window_keep_ratio: 0.8
  
  # 脉冲编码配置
  spike_encoder:
    type: voxel  # 可改为其他编码方式
```

### 5.3 添加新模块的步骤

1. **创建模块文件**
   ```python
   # src/models/modules/attention/my_attention.py
   
   @register_module("attention", "my_attention")
   class MyAttention(nn.Module):
       def forward(self, x):
           # 你的实现
           return output
   ```

2. **在配置中使用**
   ```yaml
   model:
     attention:
       type: my_attention
       my_param: 123
   ```

3. **注册模块**
   ```python
   # src/models/modules/attention/__init__.py
   from .my_attention import MyAttention
   ```

---

## 6. 软硬件协同优化建议

### 6.1 硬件友好的改进方向

#### 方向1：量化
- **位置**: `configs/hw/quant_spec.yaml`
- **内容**: 权重/激活/膜电位量化
- **收益**: 减少存储和计算精度

#### 方向2：稀疏化
- **Token剪枝**: 减少计算量
- **Window剪枝**: 只计算重要区域
- **Head剪枝**: 减少注意力头

#### 方向3：算子融合
- 合并相邻的卷积层
- 融合spike操作和卷积

### 6.2 具体实现建议

#### 建议1：添加稀疏attention

```python
# 在 attention 模块中添加稀疏mask

class SparseWindowAttention(nn.Module):
    def forward(self, q, k, v, sparse_mask=None):
        attn = q * k
        
        # 应用稀疏mask
        if sparse_mask is not None:
            attn = attn * sparse_mask
        
        return self.proj(attn)
```

#### 建议2：添加硬件感知的剪枝

```python
# 基于 FPGA/ASIC 约束的剪枝

class HardwareAwarePruner:
    def __init__(self, target_device="FPGA"):
        self.block_size = 8  # FPGA友好的块大小
    
    def prune(self, x):
        # 按块剪枝，而不是按token
        blocks = x.reshape(B, T, C, H//8, 8, W//8, 8)
        block_activity = blocks.abs().mean(...)
        # ... 块级剪枝逻辑
```

#### 建议3：量化感知训练

```python
# 在训练中模拟量化误差

class QATSpikingNeuron(nn.Module):
    def forward(self, x):
        # 量化
        x_quant = self.quantize(x)
        
        # 脉冲操作
        spike = self.neuron(x_quant)
        
        # 反量化（用于梯度）
        spike_dequant = self.dequantize(spike)
        
        return spike_dequant
```

---

## 📖 学习路线建议

### 第一周：理解基础
1. 运行baseline代码，确保能训练
2. 理解数据格式和加载流程
3. 跟踪一次完整的数据流

### 第二周：深入模块
1. 理解Patch Embed的实现
2. 理解Attention机制
3. 理解Decoder和Flow Head

### 第三周：开始改进
1. 选择一个改进方向
2. 实现并测试
3. 记录实验结果

---

## 🔗 关键文件索引

| 功能 | 文件路径 |
|------|---------|
| 训练入口 | `src/trainers/train.py` |
| 模型适配器 | `src/models/sdformer/backbone.py` |
| 配置文件 | `configs/sdformer_baseline.yaml` |
| 数据加载 | `src/datasets/optical_flow_dsec.py` |
| 损失函数 | `src/trainers/losses.py` |
| 稀疏模块 | `src/models/modules/sparse_ops/` |
| 注意力模块 | `src/models/modules/attention/` |
| 脉冲神经元 | `src/models/sdformer/spiking_neurons.py` |

---

**下次讲解预告**：
- 逐行分析 Patch Embed 代码
- Attention机制的详细实现
- 如何添加自己的模块

---

*小朱，有问题随时问我！*
