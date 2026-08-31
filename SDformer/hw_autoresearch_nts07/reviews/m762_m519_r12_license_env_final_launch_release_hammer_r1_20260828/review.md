# M762｜M519 R12 license-env final release fresh hammer

## Verdict

**PASS 100/100，P0/P1/P2 = 0/0/0。** 已发布且只发布一次精确的 M519 R12 DC 启动命令；本评审没有运行 runner、DC、其他 EDA 或 license query。

M760 request 内写的是 `m761` 输出名，但 orchestrator 为本次独立复核分配了 additive `m762` 身份；request/release 字节和语义均未修改。

## 通过项

- release payload `f2b6213e...`、runner `fd53e3c2...`、R12 contract/candidate、M759、M752、R11 consumed-attempt/quarantine 及全部双 seal 均通过。
- release 为 `launch_now=true`；闭合 authorization 仅允许一次 DC，VCS/Formality/PT/PTPX/remote 全 false。
- candidate 的 authorization/identity/provenance/license/unique-attempt/resource/P2/repair 共 11 个关键 section 在 release 中逐字义相等。
- R12 canonical、attempt sentinel 及 work/preflight identity 均不存在。
- 3 次 live 样本的最小 commit headroom / MemAvailable / SwapFree 分别为 94,196,628 / 416,941,132 / 56,658,940 KiB，超过 runner 的 64/128/32 GiB 门；cgroup 计数均为 0，同 UID EDA collision 为 0。
- 另一 UID 的长期 `simv` 仅作上下文记录，不触发 runner 的 same-UID collision 门。

## License 边界

遵守 sealed M760 request 的“reviewer 不查询 license server”约束，本评审没有重新查询。因此 PASS **不声称当前 license 可用**，也不依赖当前 seat 状态。runner 必须在资源门之后、attempt sentinel 之前查询 server、`Design-Compiler` 和 `DC-Ultra`，两者均有可解析 free seat 才能继续，并双封原始查询证据。查询不等于 checkout/reservation。

## 唯一允许命令

```bash
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat M519_R12_EXPECTED_DC_RUNNER_SHA256=fd53e3c2a706bfbcc1ee43b7ccb9a7f40030593862ad2192e535b28e49d53afc M519_R12_EXPECTED_DC_LAUNCH_ADMISSION_SHA256=f2b6213e757020b469b2b2249953da474cc9819c5f039d26a116c769136b285a /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m519_r12_license_env_three_axis_exact_sha_r1.sh
```

只准调用一次，不得修改环境、SHA pin 或路径。当前没有 DC 完成、面积、时序、功耗、throughput/mm²、系统加速或 paper-PPA-ready 结论；这些必须等 R12 canonical 结果完成并经过独立 result hammer。
