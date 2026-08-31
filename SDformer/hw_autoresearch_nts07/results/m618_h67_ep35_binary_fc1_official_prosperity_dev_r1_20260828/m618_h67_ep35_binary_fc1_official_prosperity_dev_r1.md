# M618 H67 ep35 binary FC1 × 官方 Prosperity CPU 重放（开发结果）

## 结论

状态：`PASS_M618_DEV_FULL100_OFFICIAL_PROSPERITY_FC1_NOT_ADMITTED`。在冻结的 100 组 exact-binary FC1 输入上，官方 Prosperity product-sparsity 相对其同配置 bit-sparsity 的聚合周期比为 **2.372889×**；逐记录倍率 geomean/min/max 为 **2.338912× / 1.864670× / 2.735905×**。这是 external official-artifact 结果，不是本项目 RTL、全网或系统倍速。

| FC1 module | stage | density | bit cycles | product cycles | cycle ratio | support reduction |
|---|---:|---:|---:|---:|---:|---:|
| `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.fc1` | 0 | 0.164291 | 90,846,443 | 33,983,930 | 2.673218× | 69.844% |
| `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.fc1` | 0 | 0.146450 | 80,981,063 | 34,568,588 | 2.342620× | 62.910% |
| `sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.0.mlp.fc1` | 1 | 0.050834 | 28,109,270 | 11,705,522 | 2.401368× | 64.532% |
| `sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.1.mlp.fc1` | 1 | 0.081533 | 45,084,746 | 21,211,136 | 2.125522× | 59.449% |
| `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp.fc1` | 2 | 0.128482 | 71,045,780 | 28,657,136 | 2.479165× | 64.945% |
| `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.1.mlp.fc1` | 2 | 0.082087 | 45,391,016 | 24,008,576 | 1.890617× | 52.449% |
| `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp.fc1` | 2 | 0.156668 | 86,631,560 | 36,726,980 | 2.358799× | 62.500% |
| `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.3.mlp.fc1` | 2 | 0.175733 | 97,173,476 | 41,112,464 | 2.363601× | 60.820% |
| `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.4.mlp.fc1` | 2 | 0.167779 | 92,775,308 | 39,853,652 | 2.327900× | 59.633% |
| `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.5.mlp.fc1` | 2 | 0.216753 | 119,856,152 | 47,569,544 | 2.519599× | 62.885% |
| **overall** | — | 0.135289 | **757,894,814** | **319,397,528** | **2.372889×** | **63.966%** |

## 映射与边界

- 输入按冻结 `[T,B,H,W,C]` C-order little-bit 解包；官方 `run_fc` 再执行 `[T,BHW,K] -> [BHW,T,K]`，所以有效 M 行为 `b,h,w,t`，K 保持输入通道顺序。
- K=16、M=256、N=128；K/N 均整 tile，M 尾 tile 按官方 `cur_tile_size_M` 收费，没有补造激活。
- stage-3 FC1 输入非二值，未进入 M51 exact-binary 集合，故明确排除。
- 真实权重只用于 M51 SHA 身份及 K/N shape 核对；官方 product/bit CPU 路径不读取权重值，只建模 8-bit 权重流量。
- 禁止与 M481 或其他自研周期相除；禁止称 ours、full-network、PPA、energy 或 system speedup。
- 本结果在 M619 fresh hammer 前保持 development / headline_admitted=false。

## 复跑

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/pytorch310/bin/python scripts/run_m618_h67_ep35_binary_fc1_official_prosperity_iso_workload.py --workers 3
```
