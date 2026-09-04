# M504 r3 运行身份独立预审

结论：**98/100，GO_FOR_FROZEN_FULL_CPU_AUDIT_ONLY**。M504 r3 可以按三 worker 运行冻结全量 CPU 审计；本回执不授权 RTL、VCS、DC、PT/PTPX、系统倍速或 DATE headline。

## 审查对象

| 文件 | SHA-256 | 结论 |
|---|---|---|
| `system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py` | `9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e` | 匹配 |
| `contracts/m504_h67_single_port_parent_scratch_execution_contract_r3_20260827.json` | `64f1ac425520816af5250647d251c14a34e28a715723c98a50b4234b01bd9a5d` | 匹配 |
| `system_simulator/scripts/run_m504_h67_single_port_parent_scratch_r3_exact_sha.sh` | `15a0a4b6c45de15f133f76070090c086cefe6587a883ad1b28f69753ee87a0f9` | 匹配 |
| `docs/359_DATE终局冻结_20260813.md` | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` | 未变 |

## 两个运行缺陷与命名闭合

1. 相对路径缺陷已修：`args.contract.resolve()` 在读取合同和 `relative_to(ROOT)` 身份序列化之前执行。对 runner 的仓库相对参数做了小型路径检查，结果稳定为 r3 合同的仓库相对路径。
2. Python `false` 缺陷已修：源码可编译，AST 中没有名为 `false` 的 `Name`；结果字典三个字段均使用 `False`。
3. r3 身份一致：合同 schema、结果 schema、输出目录、结果 JSON 与 CSV basename 全部为 r3；没有残留的 M504 r1/r2 结果名。

## r2 调度语义没有变化

这是本次最关键的独立检查。将当前 r3 源码中的十处运行/身份替换在内存中精确逆变换，重建文件 SHA 为：

`3017dbc290db06924d4f05be7346ef2c4955169afa94fb9d24287bafd353f8df`

它与 r2 独立预审冻结的 analyzer SHA 完全相同。因此 r3 没有暗改 `cleanroom_subset`、单端口调度、deadline lookahead、BFS oracle、任务排列、pipeline 方程或四个准入门。

合同的数值语义也逐项相同：冻结坐标、`sample/operator/row-chunk/partition` 顺序、周期常数、`389974420` 锚点、四个 gate、宏面积/时序数值、三 worker 上限和 claim boundary 均不变；变化仅是 r3 恢复说明、SHA/版本身份及更短的等价描述。

生产 `policy_self_test()` 复跑结果：260 个 case 全部合法，deadline 策略相对 BFS oracle 为 0 mismatch；`[1,3,5]` 仍为 oracle/work/deadline = 4/5/4。

## Fail-closed 检查

- runner 使用 `set -euo pipefail`，精确核对 analyzer 与 contract SHA，并在输出已存在时退出。
- analyzer 再次核对六个冻结输入 SHA，限制 worker 数不超过 3，并二次拒绝已有输出。
- 结果目录只在全量计算结束后以 `mkdir(..., exist_ok=False)` 建立；并发误启动最多浪费一次计算，不会覆盖已封结果。
- 当前 r3 结果目录不存在；`bash -n`、`py_compile`、严格 JSON 解析均通过。
- 未运行全量 CPU、VCS、DC、PT、PTPX 或 GPU；未修改任何生产文件。

## 非阻塞建议

analyzer 已直接要求重建 `ideal_total == SHA` 冻结的 M473 `selected.product_cycles`，当前合同 `required_anchor`、M473 selected 和常数三者也独立核成 `389974420`。若以后重开 analyzer，可再加一条显式 `selected.product_cycles == contract.required_anchor`，以增强可读性；在当前 exact-SHA runner 下它是冗余断言，不阻塞本次 GO。

最终裁决：**允许运行 M504 r3 冻结全量 CPU 审计；仅当四个门全部通过且结果再经独立 hammer 后，才允许开发对应 RTL。**
