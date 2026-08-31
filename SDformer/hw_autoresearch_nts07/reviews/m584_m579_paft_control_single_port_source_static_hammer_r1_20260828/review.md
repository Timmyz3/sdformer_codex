# M584｜M579 PAFT/control single-port product-capture source-static hammer

日期：2026-08-28  
结论：**FAIL，61/100，P0=2，P1=4，P2=2。不得据此创建 execution release 或启动 80-record 正式 CPU 重放。**

## 一、裁决

M579 的研究问题是合理的：把 PAFT 与 control 的 80 个 packed support payload 放到同一个 M505 dead-write-only 1RW 周期模型中，分别报告 arithmetic-work、local-cycle 与 PAFT/control activity increment，并禁止三列相乘。静态检查也确认：当前双封、所有合同输入 SHA、docs/359、两臂各 40 个 packed payload SHA、10x4 cohort/order 均无漂移；代码对每个 64-row task 调用一次 `M505.simulate_liveness_task(tile, False)`，bit/product/parent 算术与 DMA/tail/commit 两臂公平的主体结构成立。

但当前 source **不可执行且不保持 M528 的冻结 task stream**：

1. 仓库默认 `/usr/bin/python3` 是 3.6.8。M579 本体可 import，但 `worker_init()` 导入 M505 时立即报 `SyntaxError: future feature annotations is not defined`；即使修掉该导入，Python 3.6 的 `ProcessPoolExecutor(max_workers=None)` 签名也没有 `mp_context`。当前另外可见的 `/usr/bin/python3.12` 没有 NumPy。因此按 shebang/当前环境不存在可用 production interpreter，正式 run 会在处理第一个 record 前失败。
2. M528 的冻结数组形状为 `[sample, operator, chunk, partition]`，C-order task stream 是 `sample -> operator -> row-chunk -> partition`。M579 第 137–139 行却是 `partition -> row-chunk`，再按 append 次序进入流水周期模型。两者在每算子 20,304 个 task 上次序不同；由于 `pipeline_cycles` 计算 `max(work[i], preprocess[i+1])`，重排会改变周期，不是纯展示差异。因此它不能声称复用了“already frozen M528/M505 same task order”。

这两项任一项都足以阻止 execution candidate/release。修复后必须重新 source-static hammer；之后才可另建 launch candidate/release，再做 fresh execution hammer。

## 二、评分

| 维度 | 得分 | 满分 | 判断 |
|---|---:|---:|---|
| SHA、双封与 payload 身份 | 18 | 20 | 合同及 80 packed payload 全通过；缺 M504 直接身份 |
| spawn/import 与资源纪律 | 4 | 15 | workers 被限制到 1..3，但默认 Python 3.6 无法执行 |
| population、16-bit/64-row 与冻结次序 | 11 | 20 | 3000x432、末块 56 row 正确；task order 与 M528 相反 |
| M505 recurrence、守恒与公平 | 18 | 20 | 每 task 调 dead-only recurrence，主守恒和同模型成本结构成立 |
| PAFT/accuracy/claim 边界 | 7 | 15 | ratio 未相乘；M255 整序列回退与 evaluator 限制未传播 |
| fail-closed 输出与复跑性 | 3 | 10 | 拒绝覆盖，但无原子发布、attempt 状态及全输入终态 rehash |
| **总分** | **61** | **100** | **FAIL** |

## 三、P0 findings

### P0-1｜当前 production Python 环境无法 spawn/import

- `/usr/bin/python3 --version`：`Python 3.6.8`。
- M579 模块 import：PASS。
- `M579.worker_init()`：FAIL，M505 第 11 行 `from __future__ import annotations` 在 Python 3.6 不存在。
- Python 3.6 `ProcessPoolExecutor` 签名只有 `(max_workers=None)`；M579 第 370 行使用 `mp_context` 和 `initializer`，同样不兼容。
- `/usr/bin/python3.12` 存在，但 `import numpy` 为 `ModuleNotFoundError`。

要求：二选一并双封。

1. 真正兼容 Python 3.6：不经 3.7+ 语法导入 M505/M504，并改用 3.6 可用的 spawn worker API；或
2. 冻结一个实际存在、包含 NumPy、支持 `mp_context` 的 Python 解释器及环境身份，并让 runner 用绝对路径调用，禁止依赖 shebang/PATH。

修后必须至少做 pre-attempt spawn/import self-test，且不得创建正式 result/attempt。

### P0-2｜M579 把 M528 的 chunk-major task stream 改成 partition-major

冻结 M528/M505：

```text
shape = [sample, operator, chunk(47), partition(432)]
flatten = sample -> operator -> chunk -> partition
```

M579 当前：

```text
for partition in range(432):
    for start in range(0, 3000, 64):
        append(task)
```

即 `sample -> operator -> partition -> chunk`。独立机械示例的前 12 个 task 分别是 `(chunk=0, partition=0..11)` 与 `(chunk=0..11, partition=0)`，不相等；简单六 task 流水例中同一 per-task 成本仅重排就从 46 cycle 变 55 cycle。

要求：用与 M528 相同的 `[operator, chunk, partition]` 物化/flatten 顺序，或构建同形状数组后 C-order flatten；加入显式顺序 assertion，并用冻结 M528 H67 ep35 行账本做小规模 task-order anchor self-test。不能只改合同文字把新次序称为 frozen same-ledger。

## 四、P1 findings

### P1-1｜M505 的传递依赖 M504 未直接锁定

M579 锁了 M505 SHA，但 M505 在 import 时动态加载 `analyze_m504_h67_single_port_parent_scratch.py`。M579 合同和硬编码均未锁 M504 SHA。M528 自身显式锁过 M505 与 M504 两个 SHA；M579 应保持同一纪律。

要求：execution input 与 source constants 都加入 M504 路径/SHA `9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e`，在 worker import 前验证。

### P1-2｜M255 只被 hash，没有被解析；完整序列回退未写入输出

M579 的 `contract["inputs"]` 循环只给 M255 做文件 SHA。结果 payload 没有读取或传播 M255 的关键反证：

- exact hardware trace 的完整 `zurich_city_09_a` 64 帧，PAFT AEE `1.36565899929375`，control `1.351884619446875`，PAFT **退化 1.0189020311889285%**；
- 全局 valid825 是单 seed；两臂没有共同 runtime evaluator SHA；
- M255 明确禁止把一序列 hardware cycles 与全局 valid825 拼成 accuracy-performance Pareto。

M579 当前只输出 valid825 `0.5730215%` 改善，容易形成选择性披露。

要求：strict-parse M255，验证 status/admission，并把 `global_valid825`、`exact_hardware_trace_ten_frames`、`full_hardware_trace_sequence`、single-seed/evaluator 限制同时原样进入 accuracy scope；`accuracy_performance_pareto=false` 必须显式出现。

### P1-3｜正式输出不是原子发布，也没有 attempt 状态机

第 454–459 行直接创建目标目录、先写 CSV、再写 JSON，最后才复核 analyzer SHA。中途异常会在正式路径留下部分结果；源码末次 SHA 失败也已留下 CSV/JSON。analyzer 自身不创建/消费/封存 `.attempt`。

要求：未来 runner 必须先原子创建 attempt，运行到同文件系统隐藏 staging 目录，完成所有终态检查、manifest/outer seal 后再一次 rename 到正式目录；任何失败只能得到 sealed quarantine/attempt，不能留下正式 result。execution candidate 必须冻结 runner SHA 和这些状态转换。

### P1-4｜只末次 rehash analyzer，不 rehash 全部输入及 payload

运行前会 hash contract inputs，worker 解包时会逐 payload hash；但运行末尾只 rehash M579 analyzer。长时运行期间 manifest、packed payload、M43、M505/M504 或 accuracy evidence 若改变，结果仍可能发布为启动前 identities。

要求：在 staging 发布前重新验证 execution contract、analyzer、M43/M504/M505、两 manifest、80 payload、M247、M255、M528 hammer、docs359 全部 SHA；结果中写入终态 observed identity。

## 五、P2 findings

### P2-1｜容量数字正确但为未复算的 hard-code

`213,376 / 245,760 = 0.8682291667` 与 M528 result hammer 一致，余量 32,384 B；但 M579 没有 strict-parse M528 hammer/容量 ledger，也没披露 M528 的物理限制：九个 128x128b 1RW macro 的 integration/PPA/energy 仍 open。建议把 exact field checks 与 caveat 放进输出，避免把“fits 240 KiB”误读成 integrated macro PPA。

### P2-2｜paired cohort 的 record-to-cohort 语义未显式断言

冻结 manifest 本身无漂移，独立检查也确认每条 `record.sample_key/operator` 与 cohort 对应；但 `ordered_records()` 只检查整数 index 和 negative_count。建议显式 require record 的 sample key/operator/shape、packed bytes/plane offsets 与 cohort/固定几何一致，使未来 execution contract 不能只靠 status string 通过。

## 六、通过的机械检查

- source contract member SHA 与 outer seal：PASS。
- source analyzer SHA：`4b990906fa76543cbbccb9d244a26974914902e0b1ad546d1ad197e7edbaf1ee`，与合同一致。
- M43 SHA：`a4ddebf...adb1c3`，PASS。
- M505 SHA：`9d55d960...bc9aced`，PASS。
- docs/359 SHA：`dedde7ce...bdfc4`，未改。
- 八个合同输入 SHA：8/8 PASS。
- PAFT/control manifest：各 40 records，10x4 ordered keys，record/cohort 映射 PASS。
- packed payload：PAFT 40/40、control 40/40 SHA 与大小均 PASS，共 80/80，文件名均唯一。
- 每 record 432 partition x 3000 rows；432x16=6912 feature bits；47 个 row chunk，末块 56 rows。
- 每臂 recurrence task 数：40x432x47=812,160；双臂 1,624,320。源码的 dead-write-only recurrence 位于该双重循环内，结构上逐 task 调用一次。
- `product_issues = residual_popcount + exact_parent_rows`，并与 M505 `ideal_1r1w_issue_cycles` 对齐；M505 内部另有 parent-edge、read/forward、write/elision、bounded progress 断言。
- bit 与 product 使用相同 record/task 集、相同 8 output blocks、160-cycle weight DMA、2-cycle per-task tail、96,000-cycle per-sample commit；product 额外 matcher/front-end 成本没有被免除。
- `control/PAFT candidate cycle ratio`、`bit/product work ratio` 与 `bit/candidate local cycle ratio` 是不同字段；代码没有相乘，claim boundary 也把 `ratios_may_be_multiplied` 设为 false。
- source-only authorization 为 launch_now=false/max_attempts=0；正式 execution contract/result/attempt 均不存在。

## 七、允许的下一步

本次 verdict 是 **FAIL_SOURCE_STATIC**。root 可以修 source/source contract 并重封，然后请求新的 source-static hammer；不能把本 review 当作 launch admission。只有新 review 达到 P0=0、P1=0 后，才可另建 execution candidate/release，且 release 还要经 fresh independent hammer。不得运行 GPU/EDA/remote；不得修改 docs/359。
