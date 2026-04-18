# SDFormer模块详解 - 深度代码走读

**作者**: 小虾 🦐  
**日期**: 2026-03-27  
**目标**: 带小朱从代码层面理解每个模块的实现细节

---

## 目录

1. [Patch Embed详解](#1-patch-embed详解)
2. [Attention详解](#2-attention详解)
3. [Encoder Block详解](#3-encoder-block详解)
4. [Decoder详解](#4-decoder详解)
5. [Flow Head详解](#5-flow-head详解)
6. [稀疏化模块详解](#6-稀疏化模块详解)

---

## 1. Patch Embed详解

### 1.1 文件位置

```
文件: third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py
类名: MS_PED_Spiking_PatchEmbed_Conv_sfn
行号: 1726-1850
```

### 1.2 初始化参数

```python
def __init__(
    self,
    img_size=(240, 320),      # 输入图像尺寸
    patch_size=(2, 4, 4),      # patch大小 [时间, 高度, 宽度]
    in_chans=10,               # 输入通道数（时间bins）
    embed_dim=96,              # 嵌入维度
    ...
):
```

### 1.3 模块组成

```python
# 模块1: Head卷积（初步特征提取）
self.head = SpikingConvEncoderLayer(
    in_chans=2,      # 正负极性
    out_chans=48,    # embed_dim // 2
    kernel_size=3,
    stride=1,        # 不下采样
    padding=1
)

# 模块2: Conv下采样
self.conv = MS_SpikingConvEncoderLayer(
    in_chans=48,
    out_chans=96,    # embed_dim
    kernel_size=3,
    stride=2,        # 下采样2倍
    padding=1
)

# 模块3: 残差编码（2个残差块）
self.residual_encoding = MS_spiking_residual_feature_generator(
    dim=96,
    num_resblocks=2,
    cnt_fun='ADD'
)

# 模块4: Proj投影（最终下采样）
self.proj = SpikingPEDLayer(
    in_chans=96,
    out_chans=96,
    kernel_size=3,
    stride=4,       # 下采样4倍
    padding=1
)
```

### 1.4 Forward详解（逐行注释）

```python
def forward(self, x):
    # 输入: [B, 10, 2, H, W]
    # B=批大小, 10=时间bins, 2=正负极性, H=高度, W=宽度
    
    # === Step 1: 时间维度重排 ===
    if x.size(1) > self.num_bins:
        x = x[:, :self.num_bins, :, :, :]  # 截取前num_bins个bins
    
    # Permute: [B, 10, 2, H, W] → [B, 2, H, W, 10]
    event_reprs = x.permute(0, 2, 3, 4, 1)
    
    # 创建新的时间分bin表示
    # new_event_reprs: [B, num_ch, H, W, num_steps]
    new_event_reprs = torch.zeros(
        event_reprs.size(0), 
        self.num_ch,  # 2（正负极性）
        event_reprs.size(2), 
        event_reprs.size(3), 
        self.num_steps  # 5（时间步）
    ).to(event_reprs.device)
    
    # 重排循环
    for i in range(self.num_ch):  # i = 0, 1（正负极性）
        start = (i // 2) * self.num_steps  # 0, 0
        end = (i // 2 + 1) * self.num_steps  # 5, 5
        # 将正负极性分别填入对应位置
        new_event_reprs[:, i, :, :, :] = event_reprs[:, i % 2, :, :, start:end]
    
    # Permute: [B, 2, H, W, 5] → [5, B, 2, H, W]
    x = new_event_reprs.permute(4, 0, 1, 2, 3)
    
    # === Step 2: Head卷积 ===
    # [5, B, 2, H, W] → [5, B, 48, H, W]
    xs = self.head(x)
    
    # === Step 3: Conv下采样 ===
    # [5, B, 48, H, W] → [5, B, 96, H/2, W/2]
    xs = self.conv(xs)
    
    # === Step 4: 残差编码 ===
    # [5, B, 96, H/2, W/2] → [5, B, 96, H/2, W/2]（shape不变）
    out = self.residual_encoding(xs)
    
    # === Step 5: Proj投影 ===
    # [5, B, 96, H/2, W/2] → [5, B, 96, H/4, W/4]
    out = self.proj(out)
    
    # 最终输出: [T, B, C, H, W]
    return out
```

### 1.5 关键理解

**为什么时间维度要重排？**

原始输入 `[B, 10, 2, H, W]`：
- 10个时间bins，每个bin包含正负极性
- 这是一种"空间优先"的表示

重排后 `[5, B, 2, H, W]`：
- 5个时间步，每步包含正负极性
- 这是一种"时间优先"的表示（SNN需要）

**为什么用5步而不是10步？**
- 配置中 `num_steps=5`
- 10个bins分成5步，每步包含2个bins的信息

---

## 2. Attention详解

### 2.1 文件位置

```
文件: third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py
类名: Spiking_QK_WindowAttention3D
行号: 605-717
```

### 2.2 初始化参数

```python
def __init__(
    self,
    dim,                        # 输入维度
    window_size,               # 窗口大小 [T, H, W]
    pretrained_window_size,    # 预训练窗口大小
    num_heads,                 # 注意力头数
    ...
):
    self.dim = dim
    self.window_size = window_size  # [2, 9, 9]
    self.num_heads = num_heads
    head_dim = dim // num_heads
    
    # Query投影
    self.linear_q = sj_layer.Linear(dim, dim, bias=False)
    self.bn_q = SpikingNormLayer(dim, ...)
    self.sn_q = Spiking_neuron(...)
    
    # Key投影 + 位置编码
    self.linear_k = sj_layer.Linear(dim, dim, bias=False)
    self.positional_encoding = nn.Parameter(...)
    self.bn_k = SpikingNormLayer(dim, ...)
    self.sn_k = Spiking_neuron(...)
    
    # 输出投影
    self.proj = sj_layer.Linear(dim, dim)
    self.proj_bn = SpikingNormLayer(dim, ...)
    self.proj_sn = Spiking_neuron(...)
```

### 2.3 Forward详解（逐行注释）

```python
def forward(self, x, mask=None):
    # 输入: [T, B, H, W, C]
    T, B_, H, W, C = x.shape
    
    # === Step 1: 预处理 ===
    # 先过一次脉冲神经元（清理信号）
    x = self.proj_sn(x.float())
    
    # === Step 2: 生成Query ===
    q = self.linear_q(x)  # 线性投影
    
    # Batch Normalization（如果配置了）
    if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
        # Permute是为了匹配BN的维度要求
        q = self.bn_q(q.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
    
    # 脉冲化（生成脉冲表示）
    q = self.sn_q(q)
    
    # === Step 3: 生成Key + 位置编码 ===
    k = self.linear_k(x).float()
    
    # 添加位置编码（关键！）
    # positional_encoding: [1, num_heads, T*H*W, head_dim]
    positional_encoding = self.positional_encoding.reshape(T, 1, H, W, C)
    k = k + positional_encoding  # 广播加法
    
    # 脉冲化
    k = self.sn_k(k)
    
    # === Step 4: Reshape到多头格式 ===
    # q: [T, B, H, W, C] → [T, B, num_heads, H*W, head_dim]
    q = q.reshape(T, B_, self.num_heads, -1, C // self.num_heads)
    # k: [T, B, H, W, C] → [B, num_heads, T*H*W, head_dim]
    k = k.reshape(B_, self.num_heads, -1, C // self.num_heads)
    
    N = k.shape[2]  # T * H * W
    
    # === Step 5: 注意力计算（核心！）===
    # 聚合Query的特征维度
    att_token = q.sum(dim=-1, keepdim=True)  # [T, B, num_heads, H*W, 1]
    att_token = self.sn2_q(att_token)         # 再次脉冲化
    
    # Key与聚合token相乘
    # att_token: [T, B, num_heads, H*W, 1]
    # k: [B, num_heads, H*W, head_dim]
    # attn: [T, B, num_heads, H*W, head_dim]
    attn = k.mul(att_token.reshape(B_, self.num_heads, -1, 1))
    
    # Dropout
    attn = self.attn_drop(attn)
    
    # === Step 6: Reshape回来 ===
    # attn: [T, B, num_heads, H*W, head_dim]
    # → [B, num_heads, T, H, W, head_dim]
    # → [T, B, H, W, C]
    x = attn.reshape(B_, self.num_heads, T, H, W, C // self.num_heads)
    x = x.permute(2, 0, 3, 4, 1, 5).reshape(T, B_, H, W, C)
    
    # 脉冲化
    x = x.float()
    attn = self.attn_sn(x)
    
    # === Step 7: 输出投影 ===
    x = self.proj(x)
    
    if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
        x = self.proj_bn(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
    
    # Reshape到序列格式
    x = x.reshape(B_, N, C)  # N = T * H * W
    
    return x, attn
```

### 2.4 核心公式

```
Attention = K ⊙ (ΣQ)
```

其中：
- `K` = Key，shape: `[B, num_heads, N, head_dim]`
- `ΣQ` = Query沿特征维度求和，shape: `[B, num_heads, N, 1]`
- `⊙` = element-wise乘法

**对比标准Transformer**：

| 标准 | Spiking |
|------|---------|
| `softmax(QK^T/√d) V` | `K ⊙ ΣQ` |
| 全局归一化 | 无归一化 |
| O(N²)复杂度 | O(N)复杂度 |
| 连续值 | 脉冲值 |

---

## 3. Encoder Block详解

### 3.1 文件位置

```
文件: third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py
类名: MS_Spiking_SwinTransformerBlock3D
行号: 888-995
```

### 3.2 Block结构

```python
class MS_Spiking_SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size, ...):
        # 归一化层
        self.norm1 = SpikingNormLayer(dim, ...)
        
        # 注意力层
        self.attn = Spiking_QK_WindowAttention3D(
            dim, window_size, num_heads, ...
        )
        
        # MLP层
        self.norm2 = SpikingNormLayer(dim, ...)
        self.mlp = Spiking_Mlp(
            dim, 
            hidden_dim=int(dim * mlp_ratio),
            ...
        )
    
    def forward(self, x):
        # 标准Transformer Block结构
        # Attention + Residual
        shortcut = x
        x = self.norm1(x)
        x, attn = self.attn(x)
        x = shortcut + x  # 残差连接
        
        # MLP + Residual
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x  # 残差连接
        
        return x, attn
```

---

## 4. Decoder详解

### 4.1 文件位置

```
文件: third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py
类名: MS_SpikingTransposeDecoderLayer
行号: 461-600
```

### 4.2 Decoder结构

```python
class MS_SpikingTransposeDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ...):
        # 转置卷积（上采样）
        self.up = sj_layer.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1
        )
        
        # 脉冲神经元
        self.sn = Spiking_neuron(...)
        
        # 归一化
        self.bn = SpikingNormLayer(out_channels, ...)
    
    def forward(self, x, skip=None, prev_pred=None):
        # 上采样
        x = self.up(x)
        x = self.sn(x)
        
        # 融合skip连接
        if skip is not None:
            x = torch.cat([x, skip], dim=2)  # 通道拼接
        
        # 融合上一层预测
        if prev_pred is not None:
            x = torch.cat([x, prev_pred], dim=2)
        
        return x
```

---

## 5. Flow Head详解

### 5.1 文件位置

```
文件: third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py
类名: MS_SpikingPredLayer
行号: 607-700
```

### 5.2 Flow Head结构

```python
class MS_SpikingPredLayer(nn.Module):
    def __init__(self, in_channels, ...):
        # 脉冲神经元
        self.sn = Spiking_neuron(...)
        
        # 1x1卷积（预测光流）
        self.conv = sj_layer.Conv2d(
            in_channels, 2,  # 输出2通道（x和y方向光流）
            kernel_size=1
        )
    
    def forward(self, x):
        x = self.sn(x)
        x = self.conv(x)  # [T, B, 2, H, W]
        return x
```

---

## 6. 稀疏化模块详解

### 6.1 文件位置

```
文件: src/models/modules/sparse_ops/token_pruning.py
```

### 6.2 Token剪枝实现

```python
class StructuredTokenPruner(nn.Module):
    def __init__(self, keep_ratio=0.7):
        self.keep_ratio = keep_ratio
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        bsz, steps, chans, height, width = x.shape
        
        # 计算每个token的活动强度
        # 取绝对值后在通道维度求平均
        tokens = x.abs().mean(dim=2)  # [B, T, H, W]
        tokens = tokens.reshape(bsz, steps, height * width)
        
        # 选择top-k token
        keep = max(1, int(tokens.shape[-1] * self.keep_ratio))
        topk_indices = torch.topk(tokens, k=keep, dim=-1).indices
        
        # 生成mask
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_indices, value=True)
        mask_2d = mask.reshape(bsz, steps, height, width)
        
        # 应用mask
        pruned = x * mask_2d[:, :, None]  # 广播
        
        return {
            "tensor": pruned,
            "mask": mask_2d,
            "token_mask": mask_2d
        }
```

### 6.3 硬件优化改进建议

```python
class HardwareAwarePruner(nn.Module):
    """硬件感知的剪枝（块级剪枝）"""
    
    def __init__(self, keep_ratio=0.7, block_size=8):
        self.keep_ratio = keep_ratio
        self.block_size = block_size  # FPGA友好的块大小
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # 将特征图分成块
        blocks = x.reshape(
            B, T, C,
            H // self.block_size, self.block_size,
            W // self.block_size, self.block_size
        )
        
        # 计算每个块的活动强度
        block_activity = blocks.abs().mean(dim=(2, 4, 6))
        
        # 选择top-k块
        keep = int(block_activity.numel() * self.keep_ratio)
        topk_indices = torch.topk(block_activity.flatten(), k=keep).indices
        
        # 生成块级mask
        mask = torch.zeros_like(block_activity, dtype=torch.bool)
        mask.flatten()[topk_indices] = True
        
        # 应用mask
        # ...
        
        return pruned_x, mask
```

---

## 附录：代码调试技巧

### A. 跟踪数据流

```python
# 在关键位置添加print
def forward(self, x):
    print(f"Input shape: {x.shape}")
    x = self.patch_embed(x)
    print(f"After patch_embed: {x.shape}")
    x = self.encoder(x)
    print(f"After encoder: {x.shape}")
    # ...
```

### B. 检查脉冲发放率

```python
def check_spike_rate(tensor, name):
    """检查脉冲发放率"""
    spike_rate = (tensor != 0).float().mean()
    print(f"{name} spike rate: {spike_rate:.2%}")

# 使用
check_spike_rate(x, "After attention")
```

### C. 可视化特征图

```python
import matplotlib.pyplot as plt

def visualize_feature(x, title):
    """可视化特征图"""
    # x: [T, B, C, H, W]
    feature = x[0, 0, 0].detach().cpu().numpy()  # 取第一个时间步
    plt.imshow(feature)
    plt.title(title)
    plt.show()
```

---

*小朱，有问题随时问我！*
