# M748：M519 R11 最终释放独立打铁

结论：**PASS，100/100，P0/P1/P2=0**。允许按下方唯一 `env -i` 命令启动最多一次 M519 R11、DC-only、K1/K8/K1x8 三轴 setup/area attempt；runner 自身的即时三采样资源/同 UID EDA 碰撞门、原子 attempt 发布和运行期监测仍是最终权威。

## 核心核对

- `launch_now=true` release、`launch_now=false` candidate、M745、recovery contract、作者 handoff 和本次 request 的双封印全部通过；release payload SHA 为 `af268658...`。
- release 的 `authorization` 精确为一次 DC，VCS/Formality/PT/PTPX/remote 全为 false，且没有未知 key。
- release 与 candidate 的九个冻结核心段语义逐字相等：authorization、identity、fresh successor provenance、R10/R11 repair provenance、unique attempt、resource/collision gate、P2 fault boundary 和 repair provenance。
- runner `7c588b...` 通过 `bash -n`；contract 的 17/17 exact files、DC entry/wrapper/actual ELF 和两份 TSMC28 DB 均重新散列通过。M745 仍为 100/100、P0/P1/P2=0 的精确 sealed status。
- R11 canonical、attempt sentinel、work、preflight staging/reject、quarantine 和 pre-attempt failure receipt 均为空；`docs/359` 仍为 `dedde7ce...`。
- 三次 10 秒间隔实时采样均通过：同 UID EDA collision 为 0；最小 commit headroom 90,278,212 KiB，最小 MemAvailable 414,866,892 KiB，SwapFree 56,661,244 KiB，cgroup fail/under_oom/oom_kill 全为 0。

本评审没有运行 DC、VCS、Formality、PT、PTPX 或 remote。NO-EDA full-path 测试没有在最终 release 上重跑：该测试模式在 runner 中按设计切换到 `launch_now=false` candidate；M745 已封存其 candidate-stage 全路径结果。本次通过静态核对最终 release 的 exact SHA、双封印、closed authorization 和九段继承关系补齐最终释放门。

## 唯一允许命令

```bash
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 M519_R11_EXPECTED_DC_RUNNER_SHA256=7c588b1a95a0afb075de97d148b5a07bad9dc2040ab890c7eb00f6c507ff6692 M519_R11_EXPECTED_DC_LAUNCH_ADMISSION_SHA256=af2686585b85c7ed5f5bb501b31fd604f4d204c2781a580450b4c02410444830 /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m519_r11_setup_area_three_axis_exact_sha_r1.sh
```

准入仅意味着可以执行这一次 setup/area DC。当前仍没有 R11 面积、时序、功耗、能量、吞吐/mm²、完整 FC2 或系统加速结论。
