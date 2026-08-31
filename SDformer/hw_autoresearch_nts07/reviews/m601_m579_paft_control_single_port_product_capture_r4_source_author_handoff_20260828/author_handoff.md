# M601｜M579 r4 mechanical overlay + launch-now-false candidate author handoff

日期：2026-08-28  
状态：**AUTHOR_SOURCE_AND_CANDIDATE_ONLY；fresh static hammer required；formal CPU unauthorized。**

## 交付物

- r4 analyzer：`system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r4.py`
  - SHA256：`ba8fc0326b4d17f45d6db156d89b29da0894560d70d82f65ea5ae5f40b115195`
- immutable r4 runner：`system_simulator/scripts/run_m601_m579_paft_control_single_port_product_capture_r4_exact_sha.sh`
  - SHA256：`8c0fcbea21eb99d2ff740d2c710e552ee7db3c5f773221cc5579899e58ad53fe`
- r4 source contract：`contracts/m601_m579_paft_control_single_port_product_capture_source_contract_r4_20260828.json`
  - SHA256：`27e995145c91de62fe687cff7a5a34889047ca1c29fa29f517e27305101d0276`
  - member sidecar 与 outer sidecar 已双封。
- `launch_now=false` candidate：`contracts/m601_m579_paft_control_single_port_product_capture_execution_candidate_r4_20260828.json`
  - SHA256：`ff6aae0b782e08c48354c0f62739e553ed74991217c0cdbda1ad4929981d28c4`
  - member sidecar 与 outer sidecar已双封；`run_cpu=false`、`max_attempts=0`、`execution_release=false`。
- frozen M594 r3 analyzer：SHA256 `c684ac4d...83136fd2`；M598 PASS manifest SHA256
  `187157eb...a7bd8d8`，outer-seal file SHA256 `16f6fc4a...e9951d`。

## 唯一变化：关闭 M598-P2-01

1. runner 的 result、attempt、consumed、staging、quarantine staging/final 坐标统一使用 bash
   `[ -e path ] || [ -L path ]` 语义；任何已有目录项（包括 dangling symlink）均拒绝，任何 symlink 均不得作为
   regular file/directory 通过。
2. analyzer 对 execution contract、production output、terminal staging/result/CSV 使用
   `os.path.lexists()` + 显式 `is_symlink()` 拒绝；contract output 的 result/attempt/consumed 也做 no-symlink
   检查。
3. failure receipt 不再用 `Path.exists()` 混淆 dangling symlink，而是分别记录 lexists、is_symlink、
   is_directory；success/failure tree 封存前均拒绝 symlink。
4. result、quarantine、attempt consume 仍使用原 `renameat2(RENAME_NOREPLACE)`，没有把安全性降成覆盖式 `mv`。

## 冻结计算没有改变

- r4 顶层 `worker_init/spawn_probe/analyze_record` 只委托 exact-SHA M594 r3；r3 再锁 M586 r2、r1、M43、
  M504、M505。
- execution input仍为精确 15 key，只把 runtime source identity 从 M594 r3 analyzer/runner 替换为 M601 r4
  analyzer/runner；r3 analyzer 是 r4 的 hard-coded exact-SHA transitive dependency，不增加 payload/计算轴。
- 80 packed payload、10x4 cohort、`[sample,operator,row-chunk,partition]`、20,304 tasks/operator、末 chunk
  56 rows、DMA=160、tail=2、commit=96,000/sample、8 blocks 均冻结。
- M255 valid825 单 seed +0.5730215%、十帧 5 win/5 loss、完整 64 帧 PAFT 退化 1.0189020% 同列；
  `accuracy_performance_pareto=false`。
- M528 九行容量仍为 213,376 B / 245,760 B；macro integration/PPA/energy open。
- arithmetic/local-cycle/trained-activity ratio 不相乘；system/RTL/PPA/energy/headline 全 false。

## 已执行的测试

- Python strict compile、runner `bash -n`、runner mode 0755：PASS。
- final exact runner `--preflight-only`：PASS；Python/NumPy/spawn/M43/M504/M505、八 synthetic rows
  （ideal=6、liveness=8）、15-key 与零正式工件均通过。
- 临时 `launch_now=true` v4 contract 只做 validator：15/15 inputs 与 80/80 payload 重哈通过，正式 record=0；
  execution contract symlink 被 `must not be a symlink` 拒绝。
- 对 `launch_now=false` candidate 调用 `--execute` 在 pre-attempt schema gate 拒绝；result/attempt 坐标仍为空。
- 临时 dangling symlink 验证：`exists=false`、`is_symlink=true`、`lexists=true`、bash `-e OR -L=true`。
- 作者第一次临时 v4 validator 暴露 frozen r2 `__file__` 尚指 r3，发生在最终 source SHA 冻结前；已修为 r4
  exact identity，最终 validator/preflight 均通过。没有处理正式 record 或创建 canonical 工件。

## 授权边界

本 handoff 与 candidate **不构成 release**。fresh static hammer 必须得到 `score>=95、P0=0、P1=0`；之后仍需
root 另建 exact-SHA v4 true execution contract、独立 true-launch admission 与 release，才允许一次最多 3 worker
的 80-record CPU replay。当前正式 CPU/GPU/EDA/remote 均未授权。

`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

