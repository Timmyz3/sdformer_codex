# AllBinary 主线 ATLIF 硬件设计修正

## 1. 结论先行

之前 RTL 里的 `binary_atlif_unit.v` 只有一条比较逻辑：

```verilog
assign event_out = (membrane >= threshold);
```

这个单元只能称为 **阈值发放器** 或 **PSN 后事件发放 stage**，不能称为完整 ATLIF 神经元。如果论文中把它当成完整 ATLIF，会被质疑没有膜电位、没有累计、没有泄露、没有复位。

本次修正后，硬件设计应分成两层：

1. **stateless threshold emitter**：对应当前 H9 ATLIF wrapper 的 `h_seq -> threshold -> event` 路径，适合精确贴近现有 all-binary NTS/H60 推理图。
2. **stateful binary ATLIF-lite**：加入膜电位寄存器、泄露、累计、阈值比较、软复位，适合在 DATE 硬件故事里作为统一事件传播单元。

新增 RTL：

- `rtl_allbinary/binary_atlif_state_unit.v`

已通过 Icarus Verilog smoke test，覆盖无泄露累计、软复位、泄露三种行为。

## 2. 代码里 ATLIF 到底有没有“自适应”

主线 H9 all-binary 使用的是：

```text
neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py
```

核心 forward 逻辑是：

```python
h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
spike, thre_updates = self.act(h_seq, self.thresh, self.sp)
out = spike.view(x_seq.shape)
```

也就是说，当前主线的 ATLIF wrapper 更接近：

```text
时间维 PSN 线性混合 -> 阈值发放 -> binary/ternary event
```

这里的“自适应”主要发生在训练或校准阶段：

- `self.thresh` 是可训练阈值参数。
- `thre_updates` 会累计到 `update_value`。
- `sparsity_eta / sp` 控制稀疏正则。
- `target_rate / target_rate_eta` 可以控制目标 firing rate。
- `quantile_q` 和 importance 相关参数用于阈值保护或统计。

但在硬件推理阶段，建议把这些都视为 **离线训练/校准得到的固定参数**，不要宣称硬件里有在线 adaptive threshold，除非我们额外实现在线控制器并做稳定性验证。

## 3. 当前主线 ATLIF 没有经典 LIF 膜电位递推

当前 H9 wrapper 没有以下经典逐时刻公式：

```text
mem[t] = leak * mem[t-1] + input[t]
spike[t] = mem[t] >= threshold
mem[t] = reset(mem[t], spike[t])
```

仓库里另一个早期候选 ATLIF 才有这种形式：

```text
neuron_experiments/F2_fused_lmh_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/single/atlif.py
```

其核心逻辑是：

```python
mem = mem * self.tau + x[t]
spike01 = SpikeFn.apply(mem, self.thresh, self.lens)
spike = spike01 * self.thresh
mem = (1.0 - spike / self.thresh.detach().clamp_min(1e-6)) * mem
```

因此，严谨说法是：

- 软件主线 all-binary NTS/H60 的 ATLIF wrapper 是 **PSN-compatible ATLIF thresholding wrapper**。
- 它有阈值训练和稀疏自适应机制。
- 它不是硬件意义上的在线累计泄露 LIF 神经元。
- 如果我们想在硬件里讲“膜电位累计/泄露/复位”，需要新增 stateful ATLIF-lite，并做和软件输出的等价性或近似性验证。

## 4. 新增 stateful binary ATLIF-lite 设计

新增单元公式：

```text
leak_term = mem >> leak_shift
leaked_mem = mem - leak_term
mem_candidate = leaked_mem + input_current
event = mem_candidate >= threshold

soft reset:
  mem_next = event ? mem_candidate - threshold : mem_candidate

hard reset:
  mem_next = event ? 0 : mem_candidate
```

这个设计的优点：

- `leak_shift = 0` 时退化为无泄露累计。
- `leak_shift > 0` 时用移位实现泄露，不需要乘法器。
- 输出只保留 `event_bit`，不把 `threshold` 值写回激活 SRAM。
- 软复位比硬复位更接近 LIF 的 residual membrane 行为。
- 适合被多个层分时复用，不需要按 PyTorch 里的 105 个 ATLIF module 实例化 105 套硬件。

## 5. “输出是阈值”如何简化

软件里 binary ATLIF 常见输出是：

```text
spike = spike01 * threshold
```

这对训练和浮点图比较方便，但硬件不应该真的把每个 spike 存成 threshold-valued activation。推荐硬件语义改成：

```text
event_bit = spike01
scale/threshold = per-layer 或 per-channel descriptor
```

后续计算有三种处理方式：

1. **fold threshold into next weight/scale**  
   把阈值或增益合并进下一层定点权重/scale，激活 SRAM 只存 1-bit event。

2. **event bit + descriptor lookup**  
   数据流传 1-bit，进入需要数值幅度的 MAC 或 gated-K 单元时再读取 layer/channel scale。

3. **attention Q/K 纯事件路径**  
   H60 Q/K consensus score 只需要 binary event overlap、active count、mismatch count，完全不需要 threshold-valued spike。

这也是 all-binary 主线比 ternary 更好讲硬件的关键点：主数据流可以是 1-bit event，阈值只作为发放和缩放元数据存在。

## 6. DATE 论文里建议怎么讲 ATLIF

建议不要把贡献点写成“我们实现了复杂自适应 ATLIF”。更稳的说法是：

> We decouple ATLIF training-time threshold adaptation from inference-time event emission. The accelerator stores binary events and calibrated threshold descriptors, and reuses a stateful ATLIF-lite unit only where temporal accumulation is required.

中文内部表述：

```text
训练侧：ATLIF wrapper 学到阈值、稀疏率、活动率约束。
部署侧：硬件只执行定点阈值发放、可选泄露累计、1-bit event 传播。
```

这样可以避免两个风险：

- 不把训练时的 `thre_updates` 误讲成硬件在线自适应。
- 不把当前 comparator-only RTL 误讲成完整神经元。

## 7. 还需要补的最小验证

为了决定 DATE 主线到底采用“PSN 后阈值发放”还是“stateful ATLIF-lite”，建议补三类最小验证：

1. **golden vector 对齐**
   从 PyTorch 导出某一层 ATLIF wrapper 的 `x_seq / h_seq / threshold / spike`，RTL threshold emitter 对齐 `spike`。

2. **stateful ATLIF-lite 替换实验**
   在软件中用 `mem = mem - (mem >> leak_shift) + input` 近似替换 PSN 时间矩阵，看 AEE/AAE/energy 是否还能接受。

3. **threshold-valued spike folding 实验**
   验证把 `spike = spike01 * threshold` 改成 `event_bit + folded scale` 后，输出误差是否等价或近似等价。

如果这三项没有补齐，硬件论文里最好把 stateful ATLIF-lite 作为“可选执行模式/硬件泛化单元”，而把主线精确映射写成“PSN temporal transform + threshold event emitter”。

## 8. 本次 RTL 状态

新增：

- `rtl_allbinary/binary_atlif_state_unit.v`

更新：

- `tb_allbinary/tb_unibin_h60_modules.v`
- `sim_allbinary/run_iverilog.sh`

已验证：

```text
PASS: UniBin-H60 module smoke tests passed
```

覆盖行为：

- comparator threshold emitter
- stateful ATLIF 无泄露累计
- stateful ATLIF soft reset
- stateful ATLIF shift-based leakage
- H60 consensus count
- TTB empty skip
- Shiftmax gate
- gated-K
