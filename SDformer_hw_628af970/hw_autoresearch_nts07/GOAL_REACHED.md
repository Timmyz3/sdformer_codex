# 硬件 Autoresearch 目标达成

**最优方案：** 终极组合

| 指标 | 结果 | 目标 |
|------|------|------|
| 能耗 | 12.97 mJ | ≤ 22.0 mJ |
| FPS | 93（基线）/ 最优见表 | ≥ 30.0 |
| SRAM | 388 KB | ≤ 2048.0 KB |

**推荐配置：** `scripts/configs/best_config.json`

```json
{
  "skip_empty_windows": 1,
  "pe_mac": 256,
  "tx_sc_parallel": 1,
  "window_sram_kb": 256,
  "weight_buffer_kb": 128,
  "_sram": 388
}
```
