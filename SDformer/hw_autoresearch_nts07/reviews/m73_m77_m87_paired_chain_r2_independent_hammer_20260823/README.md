# M73/M77/M87 R2 独立打铁评审（2026-08-23）

## 结论

两个上一轮 P0 均已关闭，当前计数为 **P0=0、P1=4、P2=2**。

对当前“已经确认只有一个 successor、`nvidia-smi` 正常”的远端实例给出
**SCOPED GO**：允许它继续等待 GPU，随后执行 M73 train-only capture、M77
catalog、one-step PAFT smoke、同起点 no-PAFT full5 和 PAFT full5。对泛化的无人值守
重复启动仍为 **NO-GO**，因为没有 singleton lock，GPU idle 检测也会在
`nvidia-smi` 失败时 fail-open。

本结论只准入流水线执行，不准入性能。valid825 accuracy、cycle/system speedup、
DATE headline 均继续 **NO-GO**。

## 原 P0 关闭证据

1. **M73 与 M87 的 forward 配置已统一。** M73 queue 不再使用
   `hardware_order_q7q17_deploy.yml`，而是直接使用 SHA
   `86db3960...d1cbcc` 的 H67 float source YAML。M73 tracer、M77 builder、M87
   materializer、runtime PAFT loader 与 successor receipt gate 都绑定同一 SHA。
2. **长等待后的直接依赖 SHA gate 已补齐。** M73 对 tracer、M40 writer、profile、
   config、checkpoint、train/valid list 在等待前后检查；M87 对 builder、materializer、
   PAFT loader、trainer、tracer、source config、checkpoint 在等待前后及各 arm 前后检查。
   successor 从 M73 receipt 读取 manifest SHA 并对 manifest 本体重算，不再只 grep PASS。

## 独立 CPU 测试

- `bash -n`：M73 queue 与 M87 successor 均通过。
- `py_compile`：M73 tracer、M77 builder、M87 materializer 均通过。
- synthetic admitted fixture 实际调用 M87 materializer：通过。
- PAFT 与 no-PAFT 两个 full5 YAML 独立递归比较：epoch=5、seed=0、optimizer 完全相同；
  差异只在 experiment/note、`runtime.paired_arm` 与整个 `pattern_paft` arm。
- smoke 与 full5 的差异只限 experiment/note、epoch/save fields 和
  `max_train_steps=1`；没有额外 forward 配置漂移。
- shell 结构确认每个目录通过 unique partial 后原子 `mv`，最终 receipt 也用临时文件
  原子发布；失败 trap 会写 `DO_NOT_USE` / `DO_NOT_CITE` receipt。

## 剩余发现

### P1

1. **GPU idle probe fail-open 且存在 TOCTOU。** `nvidia-smi ... || true` 会把工具失败
   解释为 0 个计算进程；最后一次 probe 与训练进程创建之间也没有资源预留。
2. **无 singleton lock。** 两个 successor 可同时通过目录不存在检查，重复训练并竞争
   `mv`/receipt。
3. **重启时既有 PASS receipt 只检查 status。** 它不会重算 manifest/catalog/contract/
   config/log/checkpoint SHA。
4. **M73 capture 结束后无直接依赖复查，且无 transitive source manifest。** 原 P0 要求的
   直接文件等待后 gate 已关闭，但运行中改动和动态 import 的 baseline/overlay 树尚未完整封存。

### P2

1. M73 在 final-directory `mv` 与 receipt `mv` 间崩溃时，重启会拒绝现有 final 目录，
   需要人工校验恢复。
2. 既有 no-PAFT control arm 的重启准入只有非空 log 与 ep4 checkpoint；缺少 arm-level
   原子完成 receipt，也没有负向确认 log 中不存在 PAFT hook。

## 运行护栏

- GPU 释放前必须确认远端只有一个 M87 successor PID；不要启动第二份。
- `nvidia-smi` 任何错误均视为 GPU 状态未知，不应继续 launch。
- 如已有 PASS receipt，先重算全部 receipt-bound artifact，不要直接依赖 status line。
- 两个 ep4 checkpoint 完成后，必须做 paired valid825 与 clean hardware-heldout replay，
  才能讨论 PAFT 的精度收益或硬件收益。

机器可读证据见 `m73_m77_m87_r2_independent_hammer.json`；可复跑入口为
`audit_m73_m77_m87_r2.py`。本评审未构造 GPU model、未启动训练、未修改生产文件。
