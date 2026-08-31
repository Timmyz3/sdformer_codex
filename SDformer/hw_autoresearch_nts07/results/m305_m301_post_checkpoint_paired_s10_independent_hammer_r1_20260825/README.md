# M305：M301 post-checkpoint 配对 S10 独立打铁评审

结论：`G4/beta48 = NO-GO`。

修复后的 beta0 r4 与 beta48 r2 使用相同 checkpoint、config、contract、wrapper、pruning module、`no_running` BN 和 10 个有序样本。16 个目标模块全部存在；beta48 的 FC1/patch 静态掩码分数分别为 `0.3510107524240042`、`0.3587612233445567`，与 M300 一致。所有被删除权重均为零，16 个权重哈希在安装后到评估结束保持稳定。

按冻结指标协议——每帧先在有效像素上计算 AEE，再对 10 帧等权平均——独立重算得到：

- beta0：`0.9602408794585982`
- beta48：`1.070772168965247`
- ΔAEE：`+0.1105312895066488`
- 门限：`+0.02`

显示精度口径的 ΔAEE 是 `+0.11053124`。候选超出预算 `0.0905312895`，约为预算的 `5.53×`。即使改用非 headline 的全局像素加权攻击口径，ΔAEE 仍为 `+0.11044760`，结论不变。

旧 beta48 r1 无 modified-forward receipt；其 `spike_profile.json` 和 `per_frame.csv` 与旧 beta0 r3 逐字节相同。归档 Python cache 与文件时间线显示旧 wrapper 在 checkpoint 恢复前挂接掩码，随后权重被 checkpoint 覆盖，并在评估后的 runtime/M300 检查处停止。旧 r1 的 baseline AEE 不是 beta48 结果，禁止引用。

M300 的 `1.1841687216×` 仍只是 ideal sensitivity，没有 same-population/executable cycles、RTL、PPA 或 system-speedup 准入。beta32 的 ideal sensitivity 仅 `1.0503984132×`，低于 `1.15×` 门限；依据 `stop_if_s10_fails`，无需继续 beta32、valid825、cycle adapter 或 RTL 晋级实验。

评分：证据 `96/100`，硬件准入 `10/100`；`P0=0, P1=2, P2=3`。若引用旧 r1 为 beta48 精度，则会形成条件性 P0 claim error。
