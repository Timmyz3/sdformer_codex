# M77/M87 H67 PAFT 自动链独立预检 R1（2026-08-23）

## 结论

**当前自动 successor 为 NO-GO；建议在 M73 仍等待 GPU 时停止 successor，并暂停当前 M73 配置。** 本轮没有启动训练、没有构造模型、没有占用 GPU，也没有改动生产文件。

PAFT 的 hard-support STE、revoked catalog denylist 和 external admission loader 的定向机制证据仍成立；真正阻塞自动五轮训练的是两个新的 P0：

1. M73 catalog 采集使用 `hardware_order_q7q17_deploy` 配置，而 M87 从普通 H67 配置生成训练配置。前者为 `hardware_quant_enabled=true / hardware_rtl_shiftmax_enabled=true / alpha0=0.015625`，后者为 `false / false / 0.02`。四个 PAFT Conv 输入处在 attention 之后；没有逐 bit support 等价回执时，不能把前一种 forward 的 catalog 用于后一种训练。
2. successor 是延迟执行的长等待 shell，却没有在等待结束后校验 builder、materializer、source YAML、`pattern_paft.py` 和 `train.py` 的受审 SHA。另一 Codex/Grok session 在等待期修改这些路径后，现进程仍会自动启动不同代码的训练。

所以这里的 NO-GO 不是说 PAFT 方向无效，而是说“当前排队脚本一旦 GPU 空闲便自动跑 smoke+full5”不安全。对齐 forward、加 SHA gate 后，可以先放行 real one-step smoke 和候选五轮；任何 PAFT 性能/准确率结论仍要等 paired baseline、valid825 与 cycle replay。

## 可保留的正向证据

- 当前 `pattern_paft.py` SHA 为 `d3eac645...a066`，与 M75 r6 receipt 绑定的一致。
- revoked M71 catalog 在旧 override 下仍被拒绝。
- hard-support proxy 的定向 fixture 为 `8 -> 4`，只证明算术机制，不是模型或系统 2x。
- M77 builder 使用 train-only M73 schema、四个固定 operator、filtered q16 Hamming Lloyd，并对 M43 与 builder start/end 做 fail-closed 检查；在输入 trace 与训练 forward 对齐之后，builder 算术可 conditional GO。

## P1/P2

- successor 只 `grep PASS_M73_CAPTURE`，没有用 `sha256sum -c` 核验 receipt 中的 manifest 身份。
- `set -e` 失败时不写失败 receipt；半成品目录又会使重启在开头被拒绝。
- 只跑 PAFT full5，没有相同 checkpoint/seed/data order/epoch/optimizer 的 no-PAFT paired baseline，无法区分 pattern 收益、普通续训和活动塌缩。
- GPU idle probe 不构成资源锁，smoke 与 full5 之间没有 ownership/recheck。
- materializer 独立运行时不 pin SOURCE YAML SHA。

## 最小修复与重启门

1. 让 M73 用与正式 PAFT 完全相同的 forward YAML 采集；如果仍想用 hardware-order catalog，则必须先对 32 个 train samples 的四层 packed support 做两种配置逐 bit 0 mismatch 证明。
2. successor 在 M73 完成后、任何 Python 执行前，逐项 `sha256sum -c`：M73 tracer/manifest receipt、M77 builder、M87 materializer、source YAML、`pattern_paft.py`、`train.py`、checkpoint。materializer 自身也 pin SOURCE SHA。
3. 用临时文件加原子 rename 写成功/失败 receipt；失败 receipt 记录 stage、exit code、所有已存在输出，不自动覆盖或删除。
4. real M77 产物出来后先做无训练的 catalog 独立复算，再做 one-step positive smoke。
5. 正式实验跑 PAFT/no-PAFT paired full5；每个 epoch 记录 accuracy、event density、support popcount、PWP exact hit、correction count 和 cycle replay。valid825 只用于训练后 guardrail，不进入 catalog 或训练。

机器可复现结果见 `m77_m87_paft_chain_preflight.json`；生成器是 `audit_m77_m87_chain.py`。
