# M472 H67 官方 Prosperity 同负载周期重放

## 结论

状态：`PASS_M472_OFFICIAL_PROSPERITY_H67_ISO_WORKLOAD`。在冻结 H67 ep35 S10 四层 bottleneck Conv3x3 的 original16 二值矩阵上，官方 Prosperity product-sparsity 相对其官方 bit-sparsity 模式为 **2.459487×**。该数字只属于官方 Prosperity 配置和四层 Conv，不是 H67 全网或本项目硬件倍速。

| 范围 | density | bit cycles | product cycles | product/bit speedup | product ops 降低 | g_wgt 读取降低 |
|---|---:|---:|---:|---:|---:|---:|
| `sttmultires_unet.resblocks.0.conv1.0` | 0.153209 | 190,703,412 | 78,728,898 | 2.422280× | 62.466% | 62.466% |
| `sttmultires_unet.resblocks.0.conv2.0` | 0.055696 | 69,381,180 | 31,043,430 | 2.234971× | 62.122% | 62.122% |
| `sttmultires_unet.resblocks.1.conv1.0` | 0.169783 | 211,324,110 | 84,765,210 | 2.493052× | 63.564% | 63.564% |
| `sttmultires_unet.resblocks.1.conv2.0` | 0.068073 | 84,779,730 | 31,602,468 | 2.682693× | 69.818% | 69.818% |
| **overall** | 0.111690 | **556,188,432** | **226,140,006** | **2.459487×** | **63.961%** | **63.961%** |

## 等价批量化证明

每个 phase 先真实调用未修改的官方 `Simulator.run_fc` CPU 路径，输出维度取一个完整的 128-lane N tile；随后按官方源码方程展开为 N=768 的六个相同 N tile，并重新计算只发生一次的初始 DRAM 延迟。6 个 mode-phase 直接 N=768 对照均为 0 mismatch。

## 证据边界

- 真实调用 Prosperity 官方未修改 CPU `Simulator.run_fc`；CUDA 仅使用 import shim，周期函数未替换。
- 输入是冻结 H67 ep35 S10 四层 bottleneck Conv3x3 的 original16 0/1 im2col 矩阵。
- product-vs-bit 是同一官方 Prosperity 配置内部的同负载对比。
- 本结果不含 ATLIF、动态 BN、attention、FC、patch embed 或全网调度。
- 官方 Prosperity 配置与 M430/M467 的资源和 cycle boundary 不同，禁止直接相除。
- 没有由本周期重放推导能量、PPA、FPS、精度或系统倍速。

## 复现

```bash
/opt/anaconda3/envs/pytorch310/bin/python scripts/run_m472_h67_official_prosperity_iso_workload.py
```
