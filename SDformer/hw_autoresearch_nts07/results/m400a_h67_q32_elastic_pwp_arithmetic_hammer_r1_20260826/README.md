# M400A — H67 q32/O4 elastic-PWP 独立算术打铁

结论：`PASS`，评分 `94/100`，`P0=0, P1=0, P2=5`。checkpoint-bound 静态算术和 32 KiB 最坏容量可接受；仅允许继续做 frozen-runtime used-center 加权，不产生性能结论。

## 全量范围

从 M41 四个 exact-SHA `I_KY_KX_O` signed-INT8 payload 和 M338 q32 nested centers，独立遍历：

- 4 operators × 432 partitions × 32 centers = 55,296 PWP center vectors；
- 每个 vector 分成 8 个 96-lane output block；
- 共 442,368 blocks、42,467,328 lanes。

M40 映射按 `feature=i*9+ky*3+kx`，每个 partition 覆盖 16 个连续 source term；6,912 项构成无重复、无遗漏的双射。

## 算术结果

所有 PWP lane 均在 signed12 范围内，实际全局范围为 `[-1089, 1059]`，最大绝对值 1089。每 lane 的 `low8 + signed high4` 重构 mismatch 为 0；block narrow 标志与“96 lanes 全部属于 `[-128,127]`”逐块等价，mismatch 为 0。

静态 narrow block 为 112,167 / 442,368，即 25.356038%。四层分别为 21.7484%、28.1594%、24.4792%、27.0372%。这是全 catalog 等权静态比例，不是 frozen runtime 使用比例，更不是流量或周期节省。

两个负控都按要求失败：把 high4 当无符号数会破坏 21,240,121 lanes、全部 442,368 blocks；漏掉 high4、只保留 signed low8 会破坏 3,057,319 lanes、330,201 blocks。

## 容量

每个 96-lane block 为 96B low8 + 48B packed high4 + 16B alignment padding = 160B。O4 每 center 固定 stride 为 640B；q32 config 为 64B patterns + 32B bitmap/control = 96B。

`96B config + 6,144B weights + 32×640B PWP = 26,720B`，放入 32 KiB slot 后余 6,048B。

## 边界

M400A 没有 runtime used-center 加权、sidecar DMA/command 账本、cycle recurrence、RTL、Synopsys、物理 SRAM、能耗或系统结果。不能把 25.3560% 直接乘到 M397 cycles，也不能称作 runtime compression 或 speedup。

## 复跑

```bash
/opt/anaconda3/bin/python \
  hw_autoresearch_nts07/results/m400a_h67_q32_elastic_pwp_arithmetic_hammer_r1_20260826/independent_arithmetic_audit_m400a.py \
  --m41 hw_autoresearch_nts07/results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/m41_h67_ep35_bottleneck_int8_bridge.json \
  --m338 hw_autoresearch_nts07/results/m338_trainonly_nested_q128_catalog_r1_20260825/m338_trainonly_nested_q128_catalog_r1.json \
  --m40 hw_autoresearch_nts07/results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/m40_bottleneck_packed_source_manifest.json \
  --docs359 hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md \
  --receipt /tmp/m400a_independent_arithmetic_receipt_r1.json
```

脚本 fail-closed，不覆盖已有 receipt。`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce…`。
